"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Convert existing partitioned data to GraphBolt format.
"""

import argparse
import importlib.metadata
import logging
import os
import sys
import time
from pathlib import Path

from dgl import distributed as dgl_distributed
from packaging import version

from graphstorm.run.launch import get_argument_parser
from graphstorm.run.launch import check_input_arguments
from graphstorm.run.launch import submit_jobs


def parse_gbconv_args() -> argparse.Namespace:
    """Parses arguments for the script"""
    parser = argparse.ArgumentParser("Convert partitioned DGL graph to GraphBolt format.")
    parser.add_argument("--input-path", type=str, required=True,
                           help="Path to input DGL partitioned data.")
    parser.add_argument("--metadata-filename", type=str, default="metadata.json",
                           help="Name for the partitioned DGL metadata file.")
    parser.add_argument("--ssh-port", type=str, default="2222", help="SSH Port")
    parser.add_argument("--ip-config", type=str,
                           help=("A file storing a list of IPs, one line for "
                                "each instance of the partition cluster."))
    parser.add_argument("--logging-level", type=str, default="info",
                           help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")
    parser.add_argument("--run-distributed", type=lambda x: (str(x).lower() in ['true', '1']),
                           help=(
                               "When set to 'true' will try to run the GraphBolt conversion in a  "
                               "distributed manner. --ip-config will need to provided. "
                               "Default: 'false'")
                           )

    return parser.parse_args()


def main():
    """ Main function
    """
    gb_conv_args = parse_gbconv_args()
    FMT = "%(asctime)s %(levelname)s %(message)s"
    # set logging level
    logging.basicConfig(
        format=FMT,
        level=getattr(logging, gb_conv_args.logging_level.upper()),
    )


    part_config = os.path.join(gb_conv_args.input_path, gb_conv_args.metadata_filename)
    dgl_version = importlib.metadata.version('dgl')
    if version.parse(dgl_version) < version.parse("2.1.0"):
        raise ValueError(
                "GraphBolt conversion requires DGL version >= 2.1.0, "
                f"but DGL version was {dgl_version}. "
            )

    gb_start = time.time()
    dgl_version = importlib.metadata.version('dgl')
    logging.info("Converting partitions to GraphBolt format")

    if gb_conv_args.run_distributed:
        if not gb_conv_args.ip_config:
            raise ValueError("IP config file path is required for distributed conversion.")
        if version.parse(dgl_version) < version.parse("2.4.0"):
            raise ValueError(
                    "GraphBolt conversion requires DGL version >= 2.4.0, "
                    f"but DGL version was {dgl_version}. "
                )
        parser = get_argument_parser()
        args, exec_script_args = parser.parse_known_args(
            sys.argv[1:] +
            ["--num-trainers", "1",
             "--part-config", part_config,
             "--ssh-port", gb_conv_args.ssh_port]
        )
        print(args)
        print(exec_script_args)
        check_input_arguments(args)

        cmd_path = str(Path(os.path.dirname(__file__)).joinpath(
            "dist_convert_to_graphbolt.py"
        ).absolute())
        print(cmd_path)

        exec_script_args = [cmd_path] + exec_script_args

        submit_jobs(args, exec_script_args)
    else:
        dgl_distributed.dgl_partition_to_graphbolt(
            part_config,
            store_eids=True,
            graph_formats="coo",
        )

    logging.info("GraphBolt conversion took %f sec.",
                    time.time() - gb_start)

if __name__ == '__main__':
    main()
