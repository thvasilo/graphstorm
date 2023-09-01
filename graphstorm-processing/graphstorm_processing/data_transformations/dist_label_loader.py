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
from dataclasses import dataclass
from typing import Dict, List
import json

from pyspark.sql import DataFrame, functions as F, SparkSession
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer

from graphstorm_processing.config.label_config_base import LabelConfig
from graphstorm_processing.data_transformations.dist_transformations import (
    DistMultiCategoryTransformation,
)


@dataclass
class SplitRates:
    """
    Dataclass to hold the split rates for each of the train/val/test splits.
    """

    train_rate: float
    val_rate: float
    test_rate: float

    def tolist(self) -> List[float]:
        """
        Return the split rates as a list of floats: [train_rate, val_rate, test_rate]
        """
        return [self.train_rate, self.val_rate, self.test_rate]


class DistLabelLoader:
    """Used to transform label columns to conform to downstream GraphStorm expectations.

    Parameters
    ----------
    label_config : LabelConfig
        A configuration object that describes the label.
    spark : SparkSession
        The SparkSession to use for processing.
    """

    def __init__(self, label_config: LabelConfig, spark: SparkSession) -> None:
        self.label_config = label_config
        self.label_column = self.label_config.cols[0]
        self.spark = spark
        self.label_map = {}  # type: Dict[str, int]

    def process_label(self, input_df: DataFrame) -> DataFrame:
        """Transforms the label column in the input DataFrame to conform to GraphStorm expectations.

        For single-label classification converts the input (String) column to a scalar (long).

        For multi-label classification converts the input (String) to a multi-hot binary vector.

        For regression the label column is unchanged, provided that it's a float.

        Parameters
        ----------
        input_df : DataFrame
            A Spark DataFrame that contains a label column that we will transform.

        Returns
        -------
        DataFrame
            A Spark DataFrame with the column label transformed.

        Raises
        ------
        RuntimeError
            If the label_config.task_type is not one of the supported task types,
            or if a passed in regression column is not of FloatType.
        """
        label_type = input_df.schema[self.label_column].dataType

        if self.label_config.task_type == "classification":
            if self.label_config.multilabel:
                assert self.label_config.separator
                label_transformer = DistMultiCategoryTransformation(
                    self.label_config.cols, self.label_config.separator
                )
                transformed_label = label_transformer.apply(input_df).select(self.label_column)
                self.label_map = label_transformer.value_map
                return transformed_label
            else:
                processed_col_name = self.label_column + "_processed"
                # TODO: We use 'keep' invalids to have missing values at index numLabels
                # This should be part of our output. We could also convert those to NaN/NULL?
                str_indexer = StringIndexer(
                    inputCol=self.label_column,
                    outputCol=processed_col_name,
                    handleInvalid="keep",
                    stringOrderType="frequencyDesc",
                )

                str_indexer_model = str_indexer.fit(input_df)
                indexed_df = str_indexer_model.transform(input_df).drop(self.label_column)

                long_class_label = indexed_df.withColumnRenamed(
                    processed_col_name, self.label_column
                ).select(F.col(self.label_column).cast("long"))

                # Get original label to encoded mapping
                label_df = self.spark.createDataFrame(
                    list(enumerate(str_indexer_model.labelsArray[0])),  # type: ignore
                    f"idx: int, {self.label_column}: string",
                )

                label_mapping = str_indexer_model.transform(label_df).select(
                    [F.col(self.label_column), F.col(processed_col_name).cast("long")]
                )
                mapping_str_list = label_mapping.toJSON().collect()
                for mapping_str in mapping_str_list:
                    map_dict = json.loads(mapping_str)
                    self.label_map[map_dict[self.label_column]] = map_dict[processed_col_name]

                return long_class_label
        elif self.label_config.task_type == "regression":
            if not isinstance(label_type, FloatType):
                raise RuntimeError(
                    "Data type for regression should be FloatType, "
                    f"got {label_type} for {self.label_column}"
                )
            return input_df.select(self.label_column)
        else:
            raise RuntimeError(
                f"Unknown label task type {self.label_config.task_type} "
                f"for type: {self.label_column}"
            )
