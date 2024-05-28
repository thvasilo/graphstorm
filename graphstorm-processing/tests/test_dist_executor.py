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
import shutil
import tempfile
from unittest import mock

import pytest

from graphstorm_processing.distributed_executor import DistributedExecutor, ExecutorConfig
from graphstorm_processing.constants import TRANSFORMATIONS_FILENAME
from test_dist_heterogenous_loader import verify_integ_test_output, NODE_CLASS_GRAPHINFO_UPDATES

pytestmark = pytest.mark.usefixtures("spark")
_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(autouse=True, name="tempdir")
def tempdir_fixture():
    """Create temp dir for output files"""
    tempdirectory = tempfile.mkdtemp(
        prefix=os.path.join(_ROOT, "resources/test_output/"),
    )
    yield tempdirectory
    shutil.rmtree(tempdirectory)


def precomp_json_file(local_input, precomp_filename):
    """Copy precomputed json to local input dir"""
    precomp_file = shutil.copy(
        os.path.join(_ROOT, "resources", "precomputed_transformations", precomp_filename),
        os.path.join(local_input, TRANSFORMATIONS_FILENAME),
    )
    return precomp_file


@pytest.fixture(name="user_state_categorical_precomp_file")
def user_state_categorical_precomp_file_fixture():
    """Copy precomputed json to local input dir"""
    precomp_file = precomp_json_file(
        os.path.join(_ROOT, "resources/small_heterogeneous_graph"),
        "user_state_categorical_transormation.json",
    )

    yield precomp_file

    os.remove(precomp_file)


def test_dist_executor_run_with_precomputed(tempdir: str, user_state_categorical_precomp_file):
    """Test run function with local data"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    executor_configuration = ExecutorConfig(
        local_config_path=input_path,
        local_output_path=tempdir,
        input_prefix=input_path,
        output_prefix=tempdir,
        num_output_files=-1,
        config_filename="gsprocessing-config.json",
        sm_execution=False,
        filesystem_type="local",
        add_reverse_edges=True,
        graph_name="small_heterogeneous_graph",
        do_repartition=True,
    )

    original_precomp_file = user_state_categorical_precomp_file

    with open(original_precomp_file, "r", encoding="utf-8") as f:
        original_transformations = json.load(f)

    dist_executor = DistributedExecutor(executor_configuration)

    # Mock the SparkContext stop() function to leave the Spark context running,
    # otherwise dist_executor stops it
    dist_executor.spark.stop = mock.MagicMock(name="stop")

    dist_executor.run()

    with open(os.path.join(tempdir, "metadata.json"), "r", encoding="utf-8") as mfile:
        metadata = json.load(mfile)

    verify_integ_test_output(metadata, dist_executor.loader, NODE_CLASS_GRAPHINFO_UPDATES)

    with open(os.path.join(tempdir, TRANSFORMATIONS_FILENAME), "r", encoding="utf-8") as f:
        reapplied_transformations = json.load(f)

    # There should be no difference between original and re-applied transformations
    assert reapplied_transformations == original_transformations

    # TODO: Verify all other metadata files have been created
