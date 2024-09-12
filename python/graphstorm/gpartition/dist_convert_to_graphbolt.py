import argparse
import concurrent.futures
import copy
import importlib.metadata
import logging
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

from dgl.base import dgl_warning
try:
    from dgl.distributed.partition import (
        _dump_part_config,
        _load_part_config,
        gb_convert_single_dgl_partition
    )
except ImportError as e:
    raise ImportError(
        "Distributed GraphBolt conversion requires DGL >= 2.4.0"
    ) from e

import graphstorm as gs
from graphstorm.config import GSConfig, get_argument_parser

def dgl_partition_to_graphbolt_distributed(
    part_config,
    store_eids=True,
    store_inner_node=False,
    store_inner_edge=False,
    graph_formats=None,
    n_jobs=1,
):
    """Convert partitions of dgl to FusedCSCSamplingGraph of GraphBolt.

    This API converts `DGLGraph` partitions to `FusedCSCSamplingGraph` which is
    dedicated for sampling in `GraphBolt`. New graphs will be stored alongside
    original graph as `fused_csc_sampling_graph.pt`.

    In the near future, partitions are supposed to be saved as
    `FusedCSCSamplingGraph` directly. At that time, this API should be deprecated.

    Parameters
    ----------
    part_config : str
        The partition configuration JSON file.
    store_eids : bool, optional
        Whether to store edge IDs in the new graph. Default: True.
    store_inner_node : bool, optional
        Whether to store inner node mask in the new graph. Default: False.
    store_inner_edge : bool, optional
        Whether to store inner edge mask in the new graph. Default: False.
    graph_formats : str or list[str], optional
        Save partitions in specified formats. It could be any combination of
        `coo`, `csc`. As `csc` format is mandatory for `FusedCSCSamplingGraph`,
        it is not necessary to specify this argument. It's mainly for
        specifying `coo` format to save edge ID mapping and destination node
        IDs. If not specified, whether to save `coo` format is determined by
        the availability of the format in DGL partitions. Default: None.
    n_jobs: int
        Number of parallel jobs to run during partition conversion. Max parallelism
        is determined by the partition count.
    """
    debug_mode = "DGL_DIST_DEBUG" in os.environ
    if debug_mode:
        dgl_warning(
            "Running in debug mode which means all attributes of DGL partitions"
            " will be saved to the new format."
        )
    part_meta = _load_part_config(part_config)
    new_part_meta = copy.deepcopy(part_meta)
    num_parts = part_meta["num_parts"]

    # [Rui] DGL partitions are always saved as homogeneous graphs even though
    # the original graph is heterogeneous. But heterogeneous information like
    # node/edge types are saved as node/edge data alongside with partitions.
    # What needs more attention is that due to the existence of HALO nodes in
    # each partition, the local node IDs are not sorted according to the node
    # types. So we fail to assign ``node_type_offset`` as required by GraphBolt.
    # But this is not a problem since such information is not used in sampling.
    # We can simply pass None to it.

    # Iterate over partitions.
    convert_with_format = partial(
        gb_convert_single_dgl_partition,
        graph_formats=graph_formats,
        part_config=part_config,
        store_eids=store_eids,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
    )
    # Need to create entirely new interpreters, because we call C++ downstream
    # See https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
    # and https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    rel_path_results = []
    if n_jobs > 1 and num_parts > 1:
        mp_ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(  # pylint: disable=unexpected-keyword-arg
            max_workers=min(num_parts, n_jobs),
            mp_context=mp_ctx,
        ) as executor:
            futures = []
            for part_id in range(num_parts):
                futures.append(executor.submit(convert_with_format, part_id))

        for part_id in range(num_parts):
            rel_path_results.append(futures[part_id].result())
    else:
        # If running single-threaded, avoid spawning new interpreter, which is slow
        for part_id in range(num_parts):
            rel_path_results.append(convert_with_format(part_id))

    for part_id in range(num_parts):
        # Update graph path.
        new_part_meta[f"part-{part_id}"][
            "part_graph_graphbolt"
        ] = rel_path_results[part_id]

    # Save dtype info into partition config.
    # [TODO][Rui] Always use int64_t for node/edge IDs in GraphBolt. See more
    # details in #7175.
    new_part_meta["node_map_dtype"] = "int64"
    new_part_meta["edge_map_dtype"] = "int64"

    _dump_part_config(part_config, new_part_meta)
    print(f"Converted partitions to GraphBolt format into {part_config}")


def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_graphbolt=config.use_graphbolt)

if __name__ == '__main__':
    arg_parser = generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_link_prediction: %s",
                    unknown_args)
    main(gs_args)
