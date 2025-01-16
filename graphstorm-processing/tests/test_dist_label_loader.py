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
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyarrow import parquet as pq
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import IntegerType, StructField, StructType, StringType


from graphstorm_processing.config.config_parser import (
    create_config_objects,
)
from graphstorm_processing.constants import DATA_SPLIT_SET_MASK_COL
from graphstorm_processing.data_transformations.dist_label_loader import SplitRates
from graphstorm_processing.graph_loaders.dist_heterogeneous_loader import (
    DistHeterogeneousGraphLoader,
    HeterogeneousLoaderConfig,
)
from graphstorm_processing.config.config_conversion import GConstructConfigConverter

from graphstorm_processing.data_transformations.dist_label_loader import DistLabelLoader
from graphstorm_processing.config.label_config_base import LabelConfig
from graphstorm_processing.constants import NODE_MAPPING_INT, NODE_MAPPING_STR

pytestmark = pytest.mark.usefixtures("spark")
_ROOT = os.path.abspath(os.path.dirname(__file__))
STR_LABEL_COL = "str_label"
NUM_LABEL_COL = "num_label"
NUM_DATAPOINTS = 10000

FORMAT_NAME = "parquet"
DELIMITER = "" if FORMAT_NAME == "parquet" else ","
CUSTOM_DATA_SPLIT_ORDER = "custom_split_order_flag"


@pytest.fixture(scope="function", name="data_configs_with_label")
def data_configs_with_label_fixture():
    """Create data configuration object that contain features and labels"""
    config_path = os.path.join(
        _ROOT, "resources/small_heterogeneous_graph/gsprocessing-config.json"
    )

    with open(config_path, "r", encoding="utf-8") as conf_file:
        gsprocessing_config = json.load(conf_file)

    data_configs_dict = create_config_objects(gsprocessing_config["graph"])

    return data_configs_dict


@pytest.fixture(scope="function", name="no_label_data_configs")
def no_label_data_configs_fixture():
    """Create data configuration object without labels"""
    config_path = os.path.join(
        _ROOT, "resources/small_heterogeneous_graph/gconstruct-no-labels-config.json"
    )

    with open(config_path, "r", encoding="utf-8") as conf_file:
        gconstruct_config = json.load(conf_file)
        gsprocessing_config = GConstructConfigConverter().convert_to_gsprocessing(gconstruct_config)

    data_configs_dict = create_config_objects(gsprocessing_config["graph"])

    return data_configs_dict


@pytest.fixture(scope="function", name="dghl_loader")
def dghl_loader_fixture(
    spark, data_configs_with_label, tmp_path: Path
) -> DistHeterogeneousGraphLoader:
    """Create a re-usable loader that includes labels"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    loader_config = HeterogeneousLoaderConfig(
        add_reverse_edges=True,
        data_configs=data_configs_with_label,
        enable_assertions=True,
        graph_name="small_heterogeneous_graph",
        input_prefix=input_path,
        local_input_path=input_path,
        local_metadata_output_path=str(tmp_path),
        num_output_files=1,
        output_prefix=str(tmp_path),
        precomputed_transformations={},
    )
    dhgl = DistHeterogeneousGraphLoader(
        spark,
        loader_config=loader_config,
    )
    return dhgl


def test_dist_classification_label(spark: SparkSession, check_df_schema):
    """Test classification label generation"""
    label_col = "name"
    classification_config = {
        "column": label_col,
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    data = [
        ("mark", "a"),
        ("john", "b"),
        ("tara", "c"),
        ("jen", "d"),
        (None, "e"),
    ]

    order_col = "ORDER_COL"
    names_df = spark.createDataFrame(data, schema=[label_col, NODE_MAPPING_STR])
    names_df = names_df.withColumn(order_col, F.monotonically_increasing_id())

    label_transformer = DistLabelLoader(LabelConfig(classification_config), spark)

    transformed_labels = label_transformer.process_label(names_df, order_col)

    label_map = label_transformer.label_map

    assert set(label_map.keys()) == {"mark", "john", "tara", "jen"}

    check_df_schema(transformed_labels.select(label_col))

    transformed_rows = transformed_labels.collect()

    row_val_set = set()
    for row in transformed_rows:
        row_val_set.add(row[label_col])

    # We expect 5 distinct label values, 4 ints and one null
    assert len(row_val_set) == 5
    for val in row_val_set:
        assert val is None or isinstance(val, int)


def test_dist_regression_label(spark: SparkSession, input_df: DataFrame, check_df_schema):
    """Test regression label generation"""
    label_col = "age"
    regression_config = {
        "column": label_col,
        "type": "regression",
        "split_rate": {"train": 1.0, "val": 0.0, "test": 0.0},
    }

    label_transformer = DistLabelLoader(LabelConfig(regression_config), spark)

    # Regression labels are expected to be float values, and we need an order col
    # and a unique ID column to process the data. In this case "name" values are unique,
    # see the definition of input_df in conftest.py
    order_col = "ORDER_COL"
    input_df = (
        input_df.withColumn(label_col, input_df[label_col].cast("float"))
        .withColumn(order_col, F.monotonically_increasing_id())
        .withColumnRenamed("name", NODE_MAPPING_STR)
    )

    transformed_labels = label_transformer.process_label(input_df, order_col)

    check_df_schema(transformed_labels.select(label_col))

    transformed_rows = transformed_labels.collect()

    row_val_set = set()
    for row in transformed_rows:
        row_val_set.add(row[label_col])

    # We expect 4 distinct label values
    assert len(row_val_set) == 4
    for val in row_val_set:
        assert isinstance(val, (float, type(None)))


def test_dist_multilabel_classification(spark: SparkSession, check_df_schema):
    """Test multilabel classification label generation"""
    label_col = "ratings"
    multilabel_config = {
        "column": label_col,
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
        "separator": "|",
    }

    data = [("1|2", 0, "a"), ("3|4", 1, "b"), ("5|6", 2, "c"), ("7|8", 3, "d"), ("NaN", 4, "e")]

    order_col = "ORDER_COL"
    schema = StructType(
        [
            StructField(label_col, StringType(), True),
            StructField(order_col, IntegerType(), False),
            StructField(NODE_MAPPING_STR, StringType(), False),
        ]
    )
    label_df = spark.createDataFrame(data, schema=schema)

    label_transformer = DistLabelLoader(LabelConfig(multilabel_config), spark)

    transformed_labels = label_transformer.process_label(label_df, order_col)

    label_map = label_transformer.label_map

    assert set(label_map.keys()) == {"1", "2", "3", "4", "5", "6", "7", "8", "NaN"}

    check_df_schema(transformed_labels.select(label_col))

    transformed_rows = transformed_labels.collect()

    row_val_list: list[list[float]] = []
    for row in transformed_rows:
        row_val_list.append(row[label_col])

    for i, row_val in enumerate(row_val_list):
        if row_val:
            assert len(row_val) == 9
            assert np.count_nonzero(row_val) == 2
            assert row_val[2 * i] == 1.0
            assert row_val[2 * i + 1] == 1.0
        else:
            assert i == 4, "Only the last row should be None/null"


def check_mask_nan_distribution(
    df: pd.DataFrame, label_column: str, mask_name: str
) -> tuple[bool, str]:
    """
    Check distribution of mask values (0/1) for NaN labels in a specific mask.

    Args:
        df: DataFrame containing labels and masks
        label_column: Name of the label column
        mask_name: Name of the mask column to check

    Returns:
        tuple: (has_error, error_message)
    """
    nan_labels = df[label_column].isna()
    mask_values = df[mask_name]

    # Count distribution of mask values for NaN labels
    nan_dist = mask_values[nan_labels].value_counts().sort_index()

    if 1 in nan_dist:
        return (
            True,
            f"Found {nan_dist[1]} NaN labels with {mask_name}=1 "
            f"(distribution of {mask_name} values for NaN labels: {nan_dist.to_dict()})",
        )
    return (False, "")


def test_dist_label_order_partitioned(
    spark: SparkSession,
    check_df_schema,
    tmp_path: Path,
):
    """Test that label and mask order is maintained after label processing"""
    label_col = "label_vals"
    id_col = NODE_MAPPING_STR
    classification_config = {
        "column": label_col,
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    # Create a Pandas DF with a label column with 10k "zero", 10k "one", 10k None rows
    num_datapoints = 10**4
    ids = list(range(3 * num_datapoints))
    data_zeros = ["zero" for _ in range(num_datapoints)]
    data_ones = ["one" for _ in range(num_datapoints)]
    data_nan = [None for _ in range(num_datapoints)]
    data = data_zeros + data_ones + data_nan
    # Create DF with label data that contains "zero", "one", None values
    # and a set of unique IDs that we treat as strings
    pandas_input = pd.DataFrame.from_dict({label_col: data, NODE_MAPPING_STR: ids})
    # We shuffle the rows so that "zero", "one" and None values are mixed and not continuous
    pandas_shuffled = pandas_input.sample(frac=1, random_state=42).reset_index(drop=True)
    # Then we assign a sequential numerical ID that we use as an order identifier
    order_col = "ORDER_COL"
    pandas_shuffled[order_col] = ids
    names_df = spark.createDataFrame(pandas_shuffled)

    # Create an order for the DF, then consistently shuffle it to multiple partitions
    # TODO: Is this deterministic? Otherwise can add a random col
    # and use as shuffle key
    names_df_repart = names_df.repartition(64)

    assert names_df_repart.rdd.getNumPartitions() == 64

    # Convert the partitioned/shuffled DF to pandas
    names_df_repart_pd = names_df_repart.toPandas()

    # Apply transformation in Spark
    label_transformer = DistLabelLoader(LabelConfig(classification_config), spark)
    transformed_labels_with_ids = label_transformer.process_label(names_df_repart, order_col)
    check_df_schema(transformed_labels_with_ids)

    label_map = label_transformer.label_map
    assert set(label_map.keys()) == {"zero", "one"}

    # Apply transformation in Pandas to check against Spark
    expected_transformed_pd = names_df_repart_pd.replace(
        {
            "zero": label_map["zero"],
            "one": label_map["one"],
        }
    )

    # Write the output DF and read it from disk to emulate real downstream scenario
    out_path = str(tmp_path.joinpath("label_order_df"))
    transformed_labels_with_ids.write.mode("overwrite").parquet(out_path)
    actual_transformed_pd: pd.DataFrame = pq.read_table(out_path).to_pandas()

    # Expect the label values to be the same, in the same order
    assert_frame_equal(
        actual_transformed_pd.loc[:, [label_col]],
        expected_transformed_pd.loc[:, [label_col]],
        check_dtype=False,
    )

    # Redundant, but let's check the values again after sorting by ID column
    assert_frame_equal(
        actual_transformed_pd.sort_values(id_col).loc[:, [label_col]],
        expected_transformed_pd.sort_values(id_col).loc[:, [label_col]],
        check_dtype=False,
    )

    # Now let's test the mask transformation
    mask_names = ("train", "val", "test")
    split_rates = SplitRates(train_rate=0.8, val_rate=0.1, test_rate=0.1)
    combined_masks = label_transformer.create_split_files_split_rates(
        transformed_labels_with_ids,
        label_col,
        order_col,
        split_rates,
        seed=42,
    )

    # Extract mask values from combined_df
    combined_masks_pandas = combined_masks.select([DATA_SPLIT_SET_MASK_COL, label_col]).toPandas()

    mask_array: np.ndarray = np.stack(combined_masks_pandas[DATA_SPLIT_SET_MASK_COL].to_numpy())
    train_mask = mask_array[:, 0].astype(np.int8)
    val_mask = mask_array[:, 1].astype(np.int8)
    test_mask = mask_array[:, 2].astype(np.int8)

    masks_and_names = zip([train_mask, val_mask, test_mask], mask_names)

    # Add mask values to the pandas DFs as individual columns
    for mask, mask_name in masks_and_names:
        combined_masks_pandas[f"{mask_name}_mask"] = mask

    # Check every mask value against the label values, and report errors
    # if there exists a mask that has value 1 in a location where the label is NaN
    errors = []
    label_values = combined_masks_pandas.loc[:, [label_col]]
    for mask_name, mask_values in masks_and_names:
        print(f"Checking {mask_name} mask")
        has_error, error_msg = check_mask_nan_distribution(label_values, label_col, mask_values)
        if has_error:
            errors.append(error_msg)

    # After groupby, raise error if any issues were found
    if errors:
        # Perform the groupby operation
        print("Grouping label values by mask values")
        grouped_values = combined_masks_pandas.groupby([label_col], dropna=False).agg(
            {i: "value_counts" for i in ["train_mask", "val_mask", "test_mask"]}
        )
        print(grouped_values)
        raise ValueError("\n".join(errors))
