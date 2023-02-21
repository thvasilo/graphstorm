""" Launch SageMaker training task
"""
import os
import argparse
import json
import logging
import boto3 # pylint: disable=import-error

import aws_utils
import s3_utils

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.s3 import S3Downloader
import sagemaker


INSTANCE_TYPE = "ml.m5.12xlarge"

def run_job(input_args, image):
    """ Run job using SageMaker estimator.PyTorch

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    """
    sm_task_name = input_args.task_name # SageMaker task name
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    instance_count = input_args.instance_count # Number of infernece instances
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm training entry_point
    num_parts = input_args.num_parts # Number of partitions
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing input graph data (unpartitioned)
    graph_data_s3 = graph_data_s3[:-1] if graph_data_s3[-1] == '/' \
        else graph_data_s3 # The input will be an S3 folder
    output_data_s3 = input_args.output_data_s3 # S3 location storing partitioned graph data
    output_data_s3 = output_data_s3[:-1] if output_data_s3[-1] == '/' \
        else output_data_s3 # The output will be an S3 folder
    metadata_filename = input_args.metadata_filename # graph metadata filename
    prefix = input_args.job_prefix

    boto_session = boto3.session.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))

    container_image_uri = image
    graph_data_s3_no_trailing = graph_data_s3[:-1] if graph_data_s3.endswith('/') else graph_data_s3

    metadata_s3_path = f"{graph_data_s3_no_trailing}/{metadata_filename}"
    metadata_local_path = os.path.join("/tmp", metadata_filename)

    if os.path.exists(metadata_local_path):
        os.remove(metadata_local_path)

    print(f"Downloading metadata file from {metadata_s3_path} into {metadata_local_path}")
    S3Downloader.download(metadata_s3_path, "/tmp/")
    with open(metadata_local_path, 'r') as meta_file: # pylint: disable=unspecified-encoding
        metadata_dict = json.load(meta_file)
        graph_name = metadata_dict["graph_name"]

    print(f"Graph name during launch: {graph_name}")
    params = {"graph-data-s3": graph_data_s3,
              "metadata-filename": metadata_filename,
              "num-parts": num_parts,
              "output-data-s3": output_data_s3,}

    print(f"Parameters {params}")

    # We split on '/' to get the bucket, as it's always the third split element in an S3 URI
    s3_input_bucket = graph_data_s3.split("/")[2]
    # Similarly, by having maxsplit=3 we get the S3 key value as the fourth element
    s3_input_key = graph_data_s3.split("/", maxsplit=3)[3]

    aws_utils.check_if_instances_available('training', instance_type, instance_count, region)

    total_byte_size = s3_utils.determine_byte_size_on_s3(
        s3_input_bucket,
        s3_input_key,
        boto_session.client(service_name="s3", region_name=region))
    input_total_size_in_gb = total_byte_size // (1024*1024*1024)
    max_allowed_volume_size = aws_utils.get_max_volume_size_for_sagemaker('training', region)
    # Heuristic, create volume size relative to input
    # Assuming compressed Parquet input, csv intermediate output
    if input_args.volume_size:
        desired_volume_size = input_args.volume_size
    else:
        desired_volume_size = 16 * (input_total_size_in_gb // instance_count)
    if desired_volume_size > max_allowed_volume_size:
        logging.warning(
            "Desired volume size (%sGB)"
            " is larger than max required, assigning the max: %sGB",
            desired_volume_size, max_allowed_volume_size)
    required_gb_per_instance = max(30, min(desired_volume_size, max_allowed_volume_size))
    logging.info(
        "Total data size: %sGB, assigning %sGB"
        " storage per instance (total storage: %sGB).",
        input_total_size_in_gb,
        required_gb_per_instance,
        instance_count*required_gb_per_instance
        )

    est = PyTorch(
        entry_point=os.path.basename(entry_point),
        source_dir=os.path.dirname(entry_point),
        image_uri=container_image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_run=input_args.max_runtime,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sagemaker_session,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Partition"}],
        container_log_level=logging.getLevelName(input_args.log_level),
        volume_size=required_gb_per_instance
    )

    est.fit(job_name=sm_task_name, wait=False)

def parse_args():
    """ Add arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-uri", type=str,
        help="Training docker image")
    parser.add_argument("--role", type=str,
        help="SageMaker role")
    parser.add_argument("--instance-type", type=str,
        default=INSTANCE_TYPE,
        help="instance type used to train models")
    parser.add_argument("--instance-count", type=int,
        default=4,
        help="number of training instances")
    parser.add_argument("--region", type=str,
        default="us-east-1",
        help="Region to launch the task")
    parser.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/scripts/sagemaker_parmetis.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_parmetis.py")
    parser.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")

    # task specific
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--metadata-filename", type=str,
        default="metadata.json", help="file name of metadata config file")
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    parser.add_argument("--output-data-s3", type=str,
        help="S3 location to store the partitioned graph")
    parser.add_argument("--volume-size", type=int, default=None)
    parser.add_argument("--job-prefix", help="Job name prefix", type=str, default='parmetis')
    parser.add_argument("--max-runtime", help="Maximum runtime in seconds", type=int, default=86400)
    parser.add_argument('--log-level', default='INFO',
        type=str, choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])

    return parser

if __name__ == "__main__":
    arg_parser = parse_args()
    args = arg_parser.parse_args()
    print(args)


    logging.basicConfig(level=args.log_level.upper())

    train_image = args.image_uri
    run_job(args, train_image)
