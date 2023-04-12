"""
A module to gather partitioning algorithm implementations to be executed in SageMaker.
"""
from typing import List, Optional, Tuple
import abc
from dataclasses import dataclass
import json
import logging
import queue
import os
import socket
import time
from threading import Thread

import psutil
import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
from sagemaker import Session

from s3_utils import upload_file_to_s3, download_data_from_s3 # pylint: disable=wrong-import-order
import utils # pylint: disable=wrong-import-order

DGL_TOOL_PATH = "/root/dgl/tools"

@dataclass()
class PartitionConfig:
    """
    Dataclass for holding the configuration for a partitioning algorithm.

    Parameters
    ----------
    metadata_file : str
        Path to the metadata file describing the graph.
    local_output_path : str
        Path to the local directory where the partitioning results should be stored.
    rank : int
        Rank of the current worker process.
    sagemaker_session : sagemaker.Session
        The SageMaker session used for training.
    """
    metadata_file: str
    local_output_path: str
    rank: int
    sagemaker_session: Session


class PartitionAlgorithm(abc.ABC):
    """
    Base class for partition algorithm implementations.

    Parameters
    ----------
    partition_config : PartitionConfig
        The configuration for the partition algorithm.
        See `PartitionConfig` for detailed argument list.
    """
    def __init__(self,
        partition_config: PartitionConfig
    ):
        self.metadata_file = partition_config.metadata_file
        self.local_output_path = partition_config.local_output_path
        self.rank = partition_config.rank
        self.sagemaker_session = partition_config.sagemaker_session

        with open(self.metadata_file, 'r', encoding='utf-8') as metafile:
            self.metadata = json.load(metafile)

        self.graph_name = self.metadata["graph_name"]

        os.makedirs(self.local_output_path, exist_ok=True)


    def create_partitions(self, output_s3_path: str, num_partitions: int) -> Tuple[str, str]:
        """
        Creates a partitioning and uploads the results to the provided S3 location.

        Expected

        Parameters
        ----------
        output_s3_path : str
            S3 prefix to upload the partitioning results to.
        num_partitions : int
            Number of partitions to create.

        Returns
        -------
        local_partition_path, s3_partition_path : Tuple[str, str]
            Paths to the local partitioning directory and S3 URI to the uploaded partition data.
        """
        local_partition_path = self._run_partitioning(num_partitions)

        # At this point the leader will need to have partition assignments and metadata
        # available locally
        if self.rank == 0:
            if not os.path.isfile(os.path.join(local_partition_path, "partition_meta.json")):
                raise RuntimeError("Expected partition_meta.json to be present in "
                    f"{local_partition_path} got contents: {os.listdir(local_partition_path)}")

        s3_partition_path = os.path.join(output_s3_path, "partition")
        self._upload_results_to_s3(local_partition_path, s3_partition_path)

        return local_partition_path, s3_partition_path

    def broadcast_partition_done(self, client_list, world_size, success=True):
        """ Notify each worker process the partition assignment process is done

        Parameters
        ----------
        client_list: list
            List of socket clients
        world_size: int
            Size of the distributed training/inference cluster
        success: bool
            True if preprocess success
        """
        if self.rank != 0:
            raise RuntimeError("broadcast_partition_done should only be called by the leader")
        msg = b"PartitionDone" if success else b"PartitionFail"
        for rank in range(1, world_size):
            client_list[rank].sendall(msg)

    def wait_for_partition_done(self, master_sock):
        """ Waiting for partition to be done

        Parameters
        ----------
        master_sock: socket
            Socket connecting master
        """
        if self.rank == 0:
            raise RuntimeError("wait_for_partition_done should only be called by a worker")
        msg = master_sock.recv(13, socket.MSG_WAITALL)
        msg = msg.decode()
        if msg != "PartitionDone":
            raise RuntimeError(f"Wait for partition Error detected, msg: {msg}")

    @abc.abstractmethod
    def _run_partitioning(self, num_partitions: int) -> str:
        """
        Runs the partitioning algorithm.

        Side-effect contract: At the end of this call the partition assignment files, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#step-1-graph-partitioning
        and a partitioning metadata JSON file, as defined in
        https://github.com/dmlc/dgl/blob/29e666152390c272e0115ce8455da1adb5fcacb1/tools/partition_algo/base.py#L8
        should exist on the leader instance (rank 0), under the returned partition_dir.

        Parameters
        ----------
        num_partitions : int
            Number of partition assignments to create.

        Returns
        -------
        partition_dir : str
            Path to the partitioning directory.
            On the leader this must contain the partition assignment data
            and a partition_meta.json file.
        """


    # TODO: Because the locations are entangled is it better if we don't take arguments here?
    @abc.abstractmethod
    def _upload_results_to_s3(self, local_partition_directory: str, output_s3_path: str) -> None:
        """
        Uploads the partitioning results to S3 once they become available on the local filesystem.

        Parameters
        ----------
        local_partition_directory : str
            Path to the partitioning directory.
        output_s3_path : str
            S3 prefix to upload the partitioning results to.
        """

class RandomPartitioner(PartitionAlgorithm): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm.
    """
    def _run_partitioning(self, num_partitions: int) -> str:
        partition_dir = os.path.join(self.local_output_path, "partition")
        os.makedirs(partition_dir, exist_ok=True)

        # Random partitioning is done on the leader node only
        if self.rank != 0:
            return partition_dir

        num_nodes_per_type = self.metadata["num_nodes_per_type"]  # type: List[int]
        ntypes = self.metadata["node_type"]  # type: List[str]
        # Note: This assumes that the order of node_type is the same as the order num_nodes_per_type
        for ntype, num_nodes_for_type in zip(ntypes, num_nodes_per_type):
            # TODO: Should be easy to allocate one ntype to each worker and distribute the work
            logging.info("Generating random partition for node type %s", ntype)
            ntype_output = os.path.join(partition_dir, f"{ntype}.txt")

            partition_assignment = np.random.randint(0, num_partitions, (num_nodes_for_type,))

            arrow_partitions = pa.Table.from_arrays(
                [pa.array(partition_assignment, type=pa.int64())],
                names=["partition_id"])
            options = pa_csv.WriteOptions(include_header=False, delimiter=' ')
            pa_csv.write_csv(arrow_partitions, ntype_output, write_options=options)

        self._create_metadata(num_partitions, partition_dir)

        return partition_dir

    @staticmethod
    def _create_metadata(num_partitions: int, partition_dir: str) -> None:
        """
        Creates the metadata file expected by the partitioning pipeline.

        https://github.com/dmlc/dgl/blob/29e666152390c272e0115ce8455da1adb5fcacb1/tools/partition_algo/base.py#L8
        defines the partition_meta.json format
        """

        partition_meta = {
            "algo_name": "random",
            "num_parts": num_partitions,
            "version": "1.0.0"
        }
        partition_meta_filepath = os.path.join(partition_dir, "partition_meta.json")
        with open(partition_meta_filepath, "w", encoding='utf-8') as metafile:
            json.dump(partition_meta, metafile)


    def _upload_results_to_s3(self, local_partition_directory: str, output_s3_path: str) -> None:
        if self.rank == 0:
            logging.debug("Uploading partition files to %s, local_partition_directory: %s",
            output_s3_path,
            local_partition_directory)
            upload_file_to_s3(output_s3_path, local_partition_directory, self.sagemaker_session)
        else:
            # Workers do not hold any partitioning information locally
            pass



@dataclass
class METISConfig:
    """
    Dataclass for holding the configuration for a METIS partitioning algorithm.

    Parameters
    ----------

    ip_list: List[str]
        List of IP addresses in the cluster.
    graph_data_path: str
        Path to where the graph data exist in the local filesystem.
    output_s3_path: str
        S3 prefix under which the intermediate METIS data will be created.
    sock: socket.socket
        Socket that the leader and workers use to communicate.
    client_list: Optional[List[socket.socket]]
        List of each worker's sockets. Only available at the leader,
        the workers will have a None value here.
    """
    ip_list: List[str]
    graph_data_path: str
    output_s3_path: str
    sock: socket.socket
    client_list: Optional[List[socket.socket]]

class METISPartitioner(PartitionAlgorithm): # pylint: disable=too-few-public-methods
    """
    Wrapper class for executing the ParMETIS partition algorithm on SageMaker,
    without access to a shared filesystem.
    """
    def __init__(self,
        partition_config: PartitionConfig,
        metis_config:  METISConfig
    ):
        super().__init__(partition_config)
        self.ip_list = metis_config.ip_list
        self.graph_data_path = metis_config.graph_data_path
        self.output_s3_path = metis_config.output_s3_path
        self.sock = metis_config.sock
        self.client_list = metis_config.client_list

        self.state_q = queue.Queue()
        self.world_size = len(metis_config.ip_list)


    @staticmethod
    def _broadcast_preprocess_done(client_list, world_size, success=True):
        """ Notify each worker process the preprocess process is done

        Parameters
        ----------
        client_list: list
            List of socket clients
        world_size: int
            Size of the distributed training/inference cluster
        success: bool
            True if preprocess success
        """
        msg = b"PreprocessDone" if success else b"PreprocessFail"
        for rank in range(1, world_size):
            client_list[rank].sendall(msg)

    @staticmethod
    def _wait_for_leader_preprocess_done(master_sock):
        """ Waiting for preprocessing done

        Parameters
        ----------
        master_sock: socket
            Socket connecting master
        """
        msg = master_sock.recv(20)
        msg = msg.decode()
        if msg != "PreprocessDone":
            raise RuntimeError(f"wait for Preprocess Error detected, msg: {msg}")

    def _launch_preprocess(self, num_parts, input_data_path,
            metadata_filename, output_path):
        """ Launch preprocessing script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        input_data_path: str
            Path to the input graph data
        meta_data_config: str
            Path to the meta data configuration
        output_path: str
            Path to store preprocess output
        """

        launch_cmd = ["mpirun", "-np", f"{num_parts}",
            "--host", ",".join(self.ip_list),
            "-wdir", f"{input_data_path}",
            "--rank-by", "node",
            "--mca", "orte_base_help_aggregate", "0",
            "/opt/conda/bin/python3", f"{DGL_TOOL_PATH}/distpartitioning/parmetis_preprocess.py",
            "--schema_file", f"{metadata_filename}",
            "--output_dir", f"{output_path}",
            "--input_dir", input_data_path,
            "--num_parts", f"{num_parts}",
            "--log_level", logging.getLevelName(logging.root.getEffectiveLevel())]

        logging.info("RUN %s", launch_cmd)
        # launch preprocess task
        thread = Thread(target=utils.run, args=(launch_cmd, self.state_q,), daemon=True)
        thread.start()
        # sleep for a while in case of ssh is rejected by peer due to busy connection
        time.sleep(0.2)
        return thread

    def _launch_parmetis(
            self,
            num_parts,
            net_ifname,
            input_data_path,
            metis_input_path):
        """ Launch parmetis script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        net_ifname: str
            Network interface used by MPI
        input_data_path: str
            Path to the input graph data
        metis_input_path: str
            Path to metis input
        """
        parmetis_nfiles = os.path.join(metis_input_path, "parmetis_nfiles.txt")
        parmetis_efiles = os.path.join(metis_input_path, "parmetis_efiles.txt")

        launch_cmd = ["mpirun", "-np", f"{num_parts}",
            "--host", ",".join(self.ip_list),
            "--rank-by", "node",
            "--mca", "orte_base_help_aggregate", "0",
            "--mca", "opal_warn_on_missing_libcuda", "0",
            "-mca", "btl_tcp_if_include", f"{net_ifname}",
            "-wdir", f"{input_data_path}",
            "-v", "/root/ParMETIS/bin/pm_dglpart3",
            f"{self.graph_name}", f"{num_parts}", f"{parmetis_nfiles}", f"{parmetis_efiles}"]

        logging.info("RUN %s", launch_cmd)
        # launch ParMetis task
        thread = Thread(target=utils.run, args=(launch_cmd, self.state_q,), daemon=True)
        thread.start()
        # sleep for a while in case of ssh is rejected by peer due to busy connection
        time.sleep(0.2)
        return thread

    def _launch_postprocess(self, meta_data_config, parmetis_output_file, partitions_dir):
        """ Launch postprocess which translates nid-partid mapping into
            Per-node-type partid mappings.

        Parameters
        ----------
        meta_data_config: str
            Path to the meta data configuration.
        parmetis_output_file: str
            Path to ParMetis output.
        partitions_dir: str
            Output path
        """
        launch_cmd = ["python3",
            f"{DGL_TOOL_PATH}/distpartitioning/parmetis_postprocess.py",
            "--postproc_input_dir", '.',
            "--schema_file", meta_data_config,
            "--parmetis_output_file", parmetis_output_file,
            "--partitions_dir", partitions_dir]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{DGL_TOOL_PATH}:{env['PYTHONPATH']}"
        logging.info("RUN %s", launch_cmd)

        # launch postprocess task
        thread = Thread(target=utils.run, args=(launch_cmd, self.state_q, env), daemon=True)
        thread.start()
        # sleep for a while in case of ssh is rejected by peer due to busy connection
        time.sleep(0.2)
        return thread

    def _leader_prepare_parmetis(self, num_partitions):
        logging.info("Leader preparing to partition graph")
        metis_input_path = os.path.join(self.local_output_path, "metis_input")
        # launch pre-processing
        preproc_task = self._launch_preprocess(num_partitions,
                                        self.graph_data_path,
                                        self.metadata_file,
                                        metis_input_path)
        preproc_task.join()
        err_code = self.state_q.get()
        if err_code != 0:
            # Notify failure to workers
            self._broadcast_preprocess_done(self.client_list, self.world_size, success=False)
            raise RuntimeError("Partition preprocessing failed")

        # Upload processed node/edge data into S3.
        # It will also upload parmetis_nfiles.txt and parmetis_efiles.txt.
        # We upload the intermediate metis_input to S3 for data backup.
        metis_input_s3 = os.path.join(self.output_s3_path, "metis_input")
        upload_file_to_s3(metis_input_s3, metis_input_path, self.sagemaker_session)

        # Upload <graph_name>_stats.txt
        # This file is generated by preprocess and consumed by parmetis
        state_file_name = f"{self.graph_name}_stats.txt"
        state_file_path = os.path.join(self.graph_data_path, state_file_name)
        upload_file_to_s3(metis_input_s3, state_file_path, self.sagemaker_session)

        # Send signal to workers that preprocess is done
        self._broadcast_preprocess_done(self.client_list, self.world_size)

        # sync on uploading metis_input data
        utils.barrier_master(self.client_list, len(self.ip_list))

        return metis_input_path

    def _leader_launch_parmetis(
            self,
            metis_input_path: str,
            num_partitions: int,
            net_ifname: str,
            partition_dir: str):
        logging.info("Leader launching ParMETIS binary and parmetis_postprocess.py")
        # launch parmetis
        metis_task = self._launch_parmetis(
            num_partitions,
            net_ifname,
            self.graph_data_path,
            metis_input_path)
        metis_task.join()
        err_code = self.state_q.get()
        if err_code != 0:
            raise RuntimeError("Parallel metis partition failed")

        # launch post processing
        parmetis_output_file = os.path.join(self.graph_data_path,
            f"{self.graph_name}_part.{num_partitions}")
        postproc_task = self._launch_postprocess(self.metadata_file,
                                            parmetis_output_file,
                                            partition_dir)
        postproc_task.join()
        err_code = self.state_q.get()
        if err_code != 0:
            self.broadcast_partition_done(self.client_list, self.world_size, success=False)
            raise RuntimeError("Post processing failed")

        # parmetis done
        logging.debug("Partition data: %s: %s",
                partition_dir, os.listdir(partition_dir))

        return partition_dir

    def _worker_prepare_parmetis(self):
        """
        Prepares the environment for workers to be able to execute the ParMETIS pipeline.

        Specifically waits to get a signal from the leader that parmetis_preprocess.py has
        finished, so it can download the required files from S3 to local filesystem.
        """
        graph_name = self.metadata["graph_name"]

        # download parmetis_nfiles.txt and parmetis_efiles.txt
        metis_input_s3 = os.path.join(self.output_s3_path, "metis_input")
        metis_input_path = os.path.join(self.local_output_path, "metis_input")
        download_data_from_s3(os.path.join(metis_input_s3, "parmetis_nfiles.txt"),
            metis_input_path,
            self.sagemaker_session)
        download_data_from_s3(os.path.join(metis_input_s3, "parmetis_efiles.txt"),
            metis_input_path,
            self.sagemaker_session)

        # Download <graph_name>_stats.txt
        # This file is generated by preprocess and consumed by parmetis
        state_file_name = f"{graph_name}_stats.txt"
        state_file_s3_path = os.path.join(metis_input_s3, state_file_name)
        download_data_from_s3(state_file_s3_path, self.graph_data_path, self.sagemaker_session)

        # we upload local metis_input
        upload_file_to_s3(metis_input_s3, metis_input_path, self.sagemaker_session)

        # done download parmetis info
        utils.barrier(self.sock)

    def _run_partitioning(self, num_partitions: int) -> str:
        partition_dir = os.path.join(self.graph_data_path, "partition")

        if self.rank == 0:  # Leader tasks
            def get_ifname():
                nics = psutil.net_if_addrs()
                for ifname, if_info in nics.items():
                    for info in if_info:
                        if info.address == self.ip_list[0] and info.family==socket.AF_INET:
                            return ifname
                raise RuntimeError("Can not find network interface")
            net_ifname = get_ifname()


            utils.barrier_master(self.client_list, self.world_size)

            metis_input_path = self._leader_prepare_parmetis(num_partitions)

            self._leader_launch_parmetis(
                metis_input_path, num_partitions, net_ifname, partition_dir)
        else:  # Worker tasks
            # TODO(thvasilo): Which master barrier does this correspond to? Is it necessary?
            utils.barrier(self.sock)

            # Wait for leader to finish ParMETIS preparation
            self._wait_for_leader_preprocess_done(self.sock)

            # Download all the necessary files locally that ParMETIS will need to run
            self._worker_prepare_parmetis()

        return partition_dir

    def _upload_results_to_s3(self, local_partition_directory: str, output_s3_path: str) -> None:
        if self.rank == 0:
            # Upload per node type partition-id into S3
            upload_file_to_s3(output_s3_path, local_partition_directory, self.sagemaker_session)
        else:
            # Workers do not hold any partitioning information
            pass
