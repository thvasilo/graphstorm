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

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyarrow import parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StructType, StringType


from graphstorm_processing.data_transformations.dist_label_loader import DistLabelLoader
from graphstorm_processing.config.label_config_base import LabelConfig
from graphstorm_processing.constants import NODE_MAPPING_INT


def test_dist_classification_label(spark: SparkSession, check_df_schema):
    """Test classification label generation"""
    label_col = "name"
    classification_config = {
        "column": "name",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    data = [
        ("mark",),
        ("john",),
        ("tara",),
        ("jen",),
        (None,),
    ]
    names_df = spark.createDataFrame(data, schema=[label_col])

    label_transformer = DistLabelLoader(LabelConfig(classification_config), spark)

    transformed_labels = label_transformer.process_label(names_df)

    label_map = label_transformer.label_map

    assert set(label_map.keys()) == {"mark", "john", "tara", "jen"}

    check_df_schema(transformed_labels)

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

    # Regression labels are expected to be float values
    input_df = input_df.withColumn(label_col, input_df[label_col].cast("float"))

    transformed_labels = label_transformer.process_label(input_df)

    check_df_schema(transformed_labels)

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

    data = [("1|2",), ("3|4",), ("5|6",), ("7|8",), ("NaN",)]

    schema = StructType([StructField("ratings", StringType(), True)])
    label_df = spark.createDataFrame(data, schema=schema)

    label_transformer = DistLabelLoader(LabelConfig(multilabel_config), spark)

    transformed_labels = label_transformer.process_label(label_df)

    label_map = label_transformer.label_map

    assert set(label_map.keys()) == {"1", "2", "3", "4", "5", "6", "7", "8", "NaN"}

    check_df_schema(transformed_labels)

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


def test_dist_label_order_partitioned(spark: SparkSession, check_df_schema, tmp_path: Path):
    """Test that label and mask order is maintained after label processing"""
    label_col = "name"
    id_col = NODE_MAPPING_INT
    classification_config = {
        "column": "name",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    num_datapoints = 10**4
    ids = list(range(3 * num_datapoints))
    data_zeros = ["one" for _ in range(num_datapoints)]
    data_ones = ["two" for _ in range(num_datapoints)]
    data_nan = [None for _ in range(num_datapoints)]
    data = data_zeros + data_ones + data_nan
    pandas_input = pd.DataFrame.from_dict({label_col: data})
    pandas_shuffled = pandas_input.sample(frac=1, random_state=42).reset_index(drop=True)
    pandas_shuffled[id_col] = ids
    print("pandas shuffled:")
    print(pandas_shuffled)
    names_df = spark.createDataFrame(pandas_shuffled)

    # Create an order for the DF, then consistently shuffle it to multiple partitions
    # names_df = names_df.withColumn("rand", F.rand(seed=54))
    # names_df = names_df.orderBy("rand")
    print("names_df no repart")
    names_df.show()
    # names_df.show()
    print(f"{names_df.rdd.getNumPartitions()=}")

    # TODO: Is this deterministic? Otherwise can add a random col
    # and use as shuffle key
    names_df_repart = names_df.repartition(64)

    assert names_df_repart.rdd.getNumPartitions() == 64

    # Convert the partitioned/shuffled DF to pandas
    names_df_repart_pd = names_df_repart.toPandas()
    print("names_df after repart:")
    names_df_repart.show()

    # Apply transformation in Spark
    label_transformer = DistLabelLoader(LabelConfig(classification_config), spark)
    transformed_labels = label_transformer.process_label(names_df_repart)
    print("transformed labels Spark DF:")
    transformed_labels.show()

    # Apply transformation in Pandas
    label_map = label_transformer.label_map
    expected_transformed_pd = names_df_repart_pd.replace(
        {
            "one": label_map["one"],
            "two": label_map["two"],
        }
    )
    print("expected transformed pandas:")
    print(expected_transformed_pd[:20])

    out_path = str(tmp_path.joinpath("label_order_df"))
    transformed_labels.write.mode("overwrite").parquet(out_path)

    check_df_schema(transformed_labels)

    print("transformed labels Pyarrow PD:")
    actual_transformed_pd: pd.DataFrame = pq.read_table(out_path).to_pandas()
    print(actual_transformed_pd[:20])

    assert set(label_map.keys()) == {"one", "two"}

    assert_frame_equal(
        actual_transformed_pd.loc[:, ["name"]],
        expected_transformed_pd.loc[:, ["name"]],
        check_dtype=False,
    )

    assert_frame_equal(
        actual_transformed_pd.sort_values(id_col).loc[:, ["name"]],
        expected_transformed_pd.sort_values(id_col).loc[:, ["name"]],
        check_dtype=False,
    )
