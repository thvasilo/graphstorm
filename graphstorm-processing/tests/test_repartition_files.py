"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import List


import pytest
from pyarrow import parquet as pq

from graphstorm_processing.repartition_files import ParquetRepartitioner
from graphstorm_processing import repartition_files

_ROOT = os.path.abspath(os.path.dirname(__file__))
DUMMY_PREFIX = "s3://dummy_bucket/dummy_prefix"


# TODO: Generate and clean up files programmatically
@pytest.fixture(scope="session", name="copy_tmp_metadata")
def copy_tmp_metadata_fixture():
    """Load and copy partition metadata file"""
    metadata_path = os.path.join(_ROOT, "resources/parquet_partitioned/partitioned_metadata.json")
    copy_path = "/tmp/partitioned_metadata.json"
    shutil.copyfile(metadata_path, copy_path)
    yield copy_path
    os.remove(copy_path)


# TODO: Add simple tests for the load functions
@pytest.fixture(autouse=True)
def tempdir():
    """Create a temporary directory for the output"""
    temp_dir = tempfile.mkdtemp(
        prefix=os.path.join(_ROOT, "resources/test_output/"),
    )
    yield temp_dir
    shutil.rmtree(temp_dir)


# This tests both partition functions, in-memory and per-file
@pytest.mark.parametrize(
    "desired_counts",
    [[10, 10, 10, 10, 10], [12, 12, 12, 9, 5], [10, 10, 15, 10, 5], [1, 1, 1, 1, 46]],
)
@pytest.mark.parametrize(
    "partition_function_name",
    ["repartition_parquet_files_in_memory", "repartition_parquet_files_streaming"],
)
def test_repartition_functions(desired_counts: List[int], partition_function_name: str):
    """Test the repartition function, streaming and in-memory"""
    assert sum(desired_counts) == 50

    my_partitioner = ParquetRepartitioner(
        os.path.join(_ROOT, "resources/parquet_partitioned"), "local"
    )

    metadata_path = os.path.join(_ROOT, "resources/parquet_partitioned/partitioned_metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as metafile:
        metadata_dict = json.load(metafile)

    edge_type_meta = metadata_dict["edges"]["src:dummy_type:dst"]

    # We have parametrized the function name since they have the same signature and result
    partition_function = getattr(my_partitioner, partition_function_name)
    updated_meta = partition_function(edge_type_meta, desired_counts)

    # Ensure we got the correct number of files, with the correct number of rows reported
    assert updated_meta["row_counts"] == desired_counts
    assert len(updated_meta["data"]) == len(desired_counts)

    # Ensure actual rows match to expectation
    for expected_count, result_filepath in zip(desired_counts, updated_meta["data"]):
        assert (
            expected_count
            == pq.read_metadata(
                os.path.join(_ROOT, "resources/parquet_partitioned", result_filepath)
            ).num_rows
        )


@pytest.mark.parametrize(
    "task_type",
    [
        "edge_class",
        "link_predict",
        pytest.param("link_prediction", marks=pytest.mark.xfail(reason="Invalid task name")),
    ],
)
def test_repartition_files_integration(monkeypatch, copy_tmp_metadata, task_type):
    """Integration test for repartition script"""
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "repartition_files.py",
                "--input-prefix",
                os.path.join(_ROOT, "resources/parquet_partitioned"),
                "--metadata-file-name",
                "partitioned_metadata.json",
                "--updated-metadata-file-name",
                "updated_row_counts_metadata.json",
            ],
        )
        # Ensure we don't accidentally modify the original metadata file
        _ = copy_tmp_metadata

        # We monkeypatch json.load to inject the task_type into the metadata
        orig_json_load = json.load

        def mock_json_load(file):
            orig_dict = orig_json_load(file)
            orig_dict["graph_info"]["task_type"] = task_type
            return orig_dict

        monkeypatch.setattr(json, "load", mock_json_load)

        # Execute main function to test side-effects
        repartition_files.main()

        # TODO: Fix location if we make this temporary
        with open("/tmp/tmp_metadata.json", "r", encoding="utf-8") as metafile:
            new_metadata_dict = json.load(metafile)

        # The most popular counts are all 10 rows
        expected_counts = [10, 10, 10, 10, 10]

        reported_edge_counts = new_metadata_dict["edges"]["src:dummy_type:dst"]["row_counts"]

        # Ensure all edge structure files have the correct reported counts
        assert reported_edge_counts == expected_counts

        try:
            edge_type = None
            for edge_type in new_metadata_dict["edge_type"]:
                edge_struct_files = new_metadata_dict["edges"][edge_type]["data"]
                # Ensure all edge structure files have the correct actual counts
                for expected_count, edge_struct_filepath in zip(expected_counts, edge_struct_files):
                    absolute_edge_filepath = os.path.join(
                        _ROOT, "resources/parquet_partitioned", edge_struct_filepath
                    )
                    assert expected_count == pq.read_metadata(absolute_edge_filepath).num_rows

                # Ensure all feature files have the correct counts, reported and actual
                for _, edge_feature_dict in new_metadata_dict["edge_data"][edge_type].items():
                    assert edge_feature_dict["row_counts"] == expected_counts
                    for expected_count, feature_filepath in zip(
                        expected_counts, edge_feature_dict["data"]
                    ):
                        absolute_feature_filepath = os.path.join(
                            _ROOT, "resources/parquet_partitioned", feature_filepath
                        )
                        assert (
                            expected_count == pq.read_metadata(absolute_feature_filepath).num_rows
                        )
        finally:
            # TODO: Move this to a cleanup fixture
            # Clean up re-partitioned structure files
            shutil.rmtree(
                Path(_ROOT)
                / Path("resources/parquet_partitioned")
                / Path(edge_struct_files[0]).parent.parent
                / Path("parquet-repartitioned")
            )
            # Clean up re-partitioned feature files
            assert edge_type, "We should have seen at least one edge type with features"
            for _, edge_feature_dict in new_metadata_dict["edge_data"][edge_type].items():
                edge_feature_path = (
                    Path(_ROOT)
                    / Path("resources/parquet_partitioned")
                    / Path(edge_feature_dict["data"][0]).parent
                )
                if "parquet-repartitioned" in edge_feature_path.parts:
                    shutil.rmtree(edge_feature_path)
            # Clean up updated metadata file
            os.remove(
                Path(_ROOT)
                / Path("resources/parquet_partitioned")
                / "updated_row_counts_metadata.json"
            )
