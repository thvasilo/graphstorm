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

import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from math import fsum
from typing import Any, Optional, Sequence

import numpy as np
from numpy.random import default_rng
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.functions import col, when, monotonically_increasing_id
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    FloatType,
)


from graphstorm_processing.config.label_config_base import LabelConfig
from graphstorm_processing.constants import (
    COLUMN_NAME,
    DATA_SPLIT_SET_MASK_COL,
    MAX_VALUE,
    MIN_VALUE,
    NODE_MAPPING_STR,
    SPECIAL_CHARACTERS,
    VALUE_COUNTS,
)
from graphstorm_processing.data_transformations.dist_transformations import (
    DistMultiLabelTransformation,
    DistSingleLabelTransformation,
)

FORMAT_NAME = "parquet"
DELIMITER = "" if FORMAT_NAME == "parquet" else ","
CUSTOM_DATA_SPLIT_ORDER = "custom_split_order_flag"


@dataclass
class SplitRates:
    """
    Dataclass to hold the split rates for each of the train/val/test splits.
    """

    train_rate: float
    val_rate: float
    test_rate: float

    def tolist(self) -> list[float]:
        """
        Return the split rates as a list of floats: [train_rate, val_rate, test_rate]
        """
        return [self.train_rate, self.val_rate, self.test_rate]

    def todict(self) -> dict[str, float]:
        """
        Return the split rates as a dict of str to float:
        {
            "train": train_rate,
            "val": val_rate,
            "test": test_rate,
        }
        """
        return {"train": self.train_rate, "val": self.val_rate, "test": self.test_rate}

    def __post_init__(self) -> None:
        """
        Validate the split rates.
        """
        # TODO: add support for sums <= 1.0, useful for large-scale link prediction
        if fsum([self.train_rate, self.val_rate, self.test_rate]) != 1.0:
            raise ValueError(
                "Sum of split rates must be 1.0, got "
                f"{self.train_rate=}, {self.val_rate=}, {self.test_rate=}"
            )


@dataclass
class CustomSplit:
    """
    Dataclass to hold the custom split for each of the train/val/test splits.

    Parameters
    ----------
    train : list[str]
        Paths of the training mask parquet files.
    valid : list[str]
        Paths of the validation mask parquet files.
    test : list[str]
        Paths of the testing mask parquet files.
    mask_columns : list[str]
        List of columns that contain original string ids.
    """

    train: list[str]
    valid: list[str]
    test: list[str]
    mask_columns: list[str]


class DistLabelLoader:
    """Used to transform label columns to conform to downstream GraphStorm expectations.

    Parameters
    ----------
    label_config : LabelConfig
        A configuration object that describes the label.
    spark : SparkSession
        The SparkSession to use for processing.
    """

    def __init__(
        self, label_config: LabelConfig, spark: SparkSession, input_prefix: Optional[str] = None
    ) -> None:
        self.label_config = label_config
        self.label_column = label_config.label_column
        self.spark = spark
        self.input_prefix = input_prefix
        self.label_map: dict[str, int] = {}

    def process_label(self, input_df_with_ids: DataFrame, order_column: str) -> DataFrame:
        """Transforms the label column in the input DataFrame to conform to GraphStorm expectations.

        For single-label classification converts the input (String) column to a scalar (long).

        For multi-label classification converts the input (String) to a multi-hot binary vector.

        For regression the label column is unchanged, provided that it's a float.

        Parameters
        ----------
        input_df_with_ids : DataFrame
            A Spark DataFrame that contains a label column that we will transform,
            and an integer ID we use to order the DataFrame.
        order_column: str
            The name of a column which we will use to maintain order in the
            input DF.

        Returns
        -------
        DataFrame
            A Spark DataFrame with the column label transformed, and
            the order column maintained.

        Raises
        ------
        RuntimeError
            If the label_config.task_type is not one of the supported task types,
            or if a passed in regression column is not of FloatType.
        """
        label_type = input_df_with_ids.schema[self.label_column].dataType

        if self.label_config.task_type == "classification":
            # TODO: SparkIndexer treats empty strings "" as valid values.
            # If we want to treat them as missing we need to pre-process the input
            if self.label_config.multilabel:
                assert self.label_config.separator
                label_transformer = DistMultiLabelTransformation(
                    [self.label_config.label_column], self.label_config.separator
                )
            else:
                label_transformer = DistSingleLabelTransformation(
                    [self.label_config.label_column], self.spark
                )

            transformed_label_with_ids = label_transformer.apply(input_df_with_ids).select(
                self.label_column, order_column
            )
            self.label_map = label_transformer.value_map
            return transformed_label_with_ids
        elif self.label_config.task_type == "regression":
            if not isinstance(label_type, FloatType):
                raise RuntimeError(
                    "Data type for regression should be FloatType, "
                    f"got {label_type} for {self.label_column}"
                )
            return input_df_with_ids.select(self.label_column, order_column)
        else:
            raise RuntimeError(
                f"Unknown label task type {self.label_config.task_type} "
                f"for type: {self.label_column}"
            )

    @staticmethod
    def _update_label_properties(
        label_properties: dict[str, Any],
        node_or_edge_type: str,
        original_labels: DataFrame,
        label_config: LabelConfig,
    ) -> None:
        """Extracts and stores statistics about labels.

        For a given node or edge type, label configuration and the original
        (non-transformed) label DataFrame, extract some statistics
        and store them in `self.label_properties` to be written to
        storage later. The statistics can be used by downstream tasks.

        Parameters
        ----------
        node_or_edge_type : str
            The node or edge type name for which the label information is being updated.
        original_labels : DataFrame
            The original (non-transformed) label DataFrame.
        label_config : LabelConfig
            The label configuration object.

        Raises
        ------
        RuntimeError
            In case an invalid task type name is specified in the label config.
        """
        label_col = label_config.label_column
        if node_or_edge_type not in label_properties:
            label_properties[node_or_edge_type] = Counter()
        # TODO: Something wrong with the assignment here? Investigate
        label_properties[node_or_edge_type][COLUMN_NAME] = label_col

        if label_config.task_type == "regression":
            label_min = original_labels.agg(F.min(label_col).alias("min")).collect()[0]["min"]
            label_max = original_labels.agg(F.max(label_col).alias("max")).collect()[0]["max"]

            current_min = label_properties[node_or_edge_type].get(MIN_VALUE, float("inf"))
            current_max = label_properties[node_or_edge_type].get(MAX_VALUE, float("-inf"))
            label_properties[node_or_edge_type][MIN_VALUE] = min(current_min, label_min)
            label_properties[node_or_edge_type][MAX_VALUE] = max(current_max, label_max)
        elif label_config.task_type == "classification":
            # Collect counts per label
            if label_config.multilabel:
                assert label_config.separator
                # Spark's split function uses a regexp so we
                # need to escape special chars to be used as separators
                if label_config.separator in SPECIAL_CHARACTERS:
                    separator = f"\\{label_config.separator}"
                else:
                    separator = label_config.separator
                label_counts_df = (
                    original_labels.select(label_col)
                    .withColumn(label_col, F.explode(F.split(F.col(label_col), separator)))
                    .groupBy(label_col)
                    .count()
                )
            else:  # Single-label classification
                label_counts_df = original_labels.groupBy(label_col).count()

            # TODO: Verify correctness here
            label_counts_dict: Counter = Counter()
            for label_count_row in label_counts_df.collect():
                row_dict = label_count_row.asDict()
                # Label value to count
                label_counts_dict[row_dict[label_col]] = row_dict["count"]

            label_properties[node_or_edge_type][VALUE_COUNTS] = (
                label_properties.get(VALUE_COUNTS, Counter()) + label_counts_dict
            )
        else:
            raise RuntimeError(f"Invalid task type: {label_config.task_type}")

    def create_split_files_split_rates(
        self,
        input_df_with_ids: DataFrame,
        label_column: str,
        order_column: str,
        split_rates: Optional[SplitRates],
        seed: Optional[int],
    ) -> DataFrame:
        """
        Creates the train/val/test mask dataframe based on split rates.

        Parameters
        ----------
        input_df_with_ids: DataFrame
            Input dataframe for which we will create split masks.
        label_column: str
            The name of the label column. If provided, the values in the column
            need to be not null for the data point to be included in one of the masks.
            If an empty string, all rows in the input_df are included in one of train/val/test sets.
        split_rates: Optional[SplitRates]
            A SplitRates object indicating the train/val/test split rates.
            If None, a default split rate of 0.8:0.1:0.1 is used.
        seed: Optional[int]
            An optional random seed for reproducibility.
        mask_field_names: Optional[tuple[str, str, str]]
            An optional tuple of field names to use for the split masks.
            If not provided, the default field names "train_mask",
            "val_mask", and "test_mask" are used.

        Returns
        -------
        spark.sql.DataFrame
            Combined dataframe that has the label column and train/val/test list column.
        """
        if split_rates is None:
            split_rates = SplitRates(train_rate=0.8, val_rate=0.1, test_rate=0.1)
            logging.info(
                "Split rate not provided for label column '%s', using split rates: %s",
                label_column,
                split_rates.tolist(),
            )
        else:
            # TODO: add support for sums <= 1.0, useful for large-scale link prediction
            if math.fsum(split_rates.tolist()) != 1.0:
                raise RuntimeError(f"Provided split rates  do not sum to 1: {split_rates}")

        split_list = split_rates.tolist()
        logging.info(
            "Creating split files for label column '%s' with split rates: %s",
            label_column,
            split_list,
        )

        rng = default_rng(seed=seed)

        # We use multinomial sampling to create a one-hot
        # vector indicating train/test/val membership
        def multinomial_sample(label_col: str) -> Sequence[int]:
            # TODO: If we guarantee the input is post-processing, we can just use -1?
            # How to handle regression in this case?
            invalid_values = {"", "None", "NaN", None, math.nan}
            if label_col in invalid_values:
                return [0, 0, 0]
            return rng.multinomial(1, split_list).astype(np.int32).tolist()

        group_col_name = DATA_SPLIT_SET_MASK_COL  # TODO: Ensure uniqueness of column?

        # TODO: Use PandasUDF and check if it is faster than UDF
        split_group = F.udf(multinomial_sample, ArrayType(IntegerType()))
        # Convert label col to string and apply UDF
        # to create one-hot vector indicating train/test/val membership
        input_col = F.col(label_column).astype("string") if label_column else F.lit("dummy")
        int_group_df = input_df_with_ids.select(
            label_column, split_group(input_col).alias(group_col_name), order_column
        )

        # Reorder and cache because we re-use this DF
        int_group_df = int_group_df.orderBy(order_column).cache()

        return int_group_df

    def create_split_files_custom_split(
        self,
        input_df: DataFrame,
        custom_split_file: CustomSplit,
        mask_field_names: Optional[tuple[str, str, str]] = None,
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        """
        Creates the train/val/test mask dataframe based on custom split files.

        Parameters
        ----------
        input_df: DataFrame
            Input dataframe for which we will create split masks.
        custom_split_file: CustomSplit
            A CustomSplit object including path to the custom split files for
            training/validation/test.
        mask_type: str
            The type of mask to create, value can be train, val or test.
        mask_field_names: Optional[tuple[str, str, str]]
            An optional tuple of field names to use for the split masks.
            If not provided, the default field names "train_mask",
            "val_mask", and "test_mask" are used.

        Returns
        -------
        tuple[DataFrame, DataFrame, DataFrame]
            Train/val/test mask dataframes.
        """

        if mask_field_names:
            mask_names = mask_field_names
        else:
            mask_names = ("train_mask", "val_mask", "test_mask")
        train_mask_df, val_mask_df, test_mask_df = (
            self._process_custom_mask_df(input_df, custom_split_file, mask_names[0], "train"),
            self._process_custom_mask_df(input_df, custom_split_file, mask_names[1], "val"),
            self._process_custom_mask_df(input_df, custom_split_file, mask_names[2], "test"),
        )
        return train_mask_df, val_mask_df, test_mask_df

    def _process_custom_mask_df(
        self, input_df: DataFrame, split_file: CustomSplit, mask_name: str, mask_type: str
    ):
        """
        Creates the mask dataframe based on custom split files on one mask type.

        Parameters
        ----------
        input_df: DataFrame
            Input dataframe for which we will add integer mapping.
        split_file: CustomSplit
            A CustomSplit object including path to the custom split files for
            training/validation/test.
        mask_name: str
            Mask field name for the mask type.
        mask_type: str
            The type of mask to create, value can be train, val or test.
        """

        def create_mapping(input_df):
            """
            Creates the integer mapping for order maintaining.

            Parameters
            ----------
            input_df: DataFrame
                Input dataframe for which we will add integer mapping.
            """
            return_df = input_df.withColumn(CUSTOM_DATA_SPLIT_ORDER, monotonically_increasing_id())
            return return_df

        if mask_type == "train":
            file_paths = split_file.train
        elif mask_type == "val":
            file_paths = split_file.valid
        elif mask_type == "test":
            file_paths = split_file.test
        else:
            raise ValueError("Unknown mask type")

        assert self.input_prefix, "Custom split files require an input prefix to read data from"

        # Custom data split should only be considered
        # in cases with a limited number of labels.
        if len(split_file.mask_columns) == 1:
            # custom split on node original id
            custom_mask_df = self.spark.read.parquet(
                *[os.path.join(self.input_prefix, file_path) for file_path in file_paths]
            ).select(col(split_file.mask_columns[0]).alias(f"custom_{mask_type}_mask"))
            input_df_id = create_mapping(input_df)
            mask_df = input_df_id.join(
                custom_mask_df,
                input_df_id[NODE_MAPPING_STR] == custom_mask_df[f"custom_{mask_type}_mask"],
                "left_outer",
            )
            mask_df = mask_df.orderBy(CUSTOM_DATA_SPLIT_ORDER)
            mask_df = mask_df.select(
                "*",
                when(mask_df[f"custom_{mask_type}_mask"].isNotNull(), 1)
                .otherwise(0)
                .alias(mask_name),
            ).select(mask_name)
        elif len(split_file.mask_columns) == 2:
            # custom split on edge (srd, dst) original ids
            custom_mask_df = self.spark.read.parquet(
                *[os.path.join(self.input_prefix, file_path) for file_path in file_paths]
            ).select(
                col(split_file.mask_columns[0]).alias(f"custom_{mask_type}_mask_src"),
                col(split_file.mask_columns[1]).alias(f"custom_{mask_type}_mask_dst"),
            )
            input_df_id = create_mapping(input_df)
            join_condition = (
                input_df_id["src_str_id"] == custom_mask_df[f"custom_{mask_type}_mask_src"]
            ) & (input_df_id["dst_str_id"] == custom_mask_df[f"custom_{mask_type}_mask_dst"])
            mask_df = input_df_id.join(custom_mask_df, join_condition, "left_outer")
            mask_df = mask_df.orderBy(CUSTOM_DATA_SPLIT_ORDER)
            mask_df = mask_df.select(
                "*",
                when(
                    (mask_df[f"custom_{mask_type}_mask_src"].isNotNull())
                    & (mask_df[f"custom_{mask_type}_mask_dst"].isNotNull()),
                    1,
                )
                .otherwise(0)
                .alias(mask_name),
            ).select(mask_name)
        else:
            raise ValueError(
                "The number of column should be only 1 or 2, got columns: "
                f"{split_file.mask_columns}"
            )

        return mask_df
