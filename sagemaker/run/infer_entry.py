"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    SageMaker inference entry point
"""
import argparse
import os

from graphstorm.config import SUPPORTED_TASKS
from graphstorm.sagemaker.sagemaker_infer import run_infer

def get_inference_parser():
    """  Add arguments for inference
    """
    parser = argparse.ArgumentParser(description='gs sagemaker train pipeline')

    parser.add_argument("--task-type", type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")

    # disrributed training
    parser.add_argument("--graph-name", type=str, help="Graph name",
        required=True)
    parser.add_argument("--graph-data-s3", type=str,
        required=True,
        help="S3 location of input training graph")
    parser.add_argument("--infer-yaml-s3", type=str,
        required=True,
        help="S3 location of inference yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--model-artifact-s3", type=str,
        required=True,
        help="S3 prefix to load the saved model artifacts from")
    parser.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    parser.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. " \
             "(Only works with node classification/regression " \
             "and edge classification/regression tasks)",
        default=None)
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom inference script provided by the user to run custom inference logic. "
             "Please provide the path of the script within the docker image")

    # following arguments are required to launch a distributed GraphStorm training task
    parser.add_argument('--data-path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--sm-dist-env', type=str, default=os.environ['SM_TRAINING_ENV'])
    parser.add_argument('--master-addr', type=str, default=os.environ['MASTER_ADDR'])
    parser.add_argument('--region', type=str, default=os.environ['AWS_REGION'])

    # Add your args if any

    return parser

if __name__ =='__main__':
    parser = get_inference_parser()
    args, unknownargs = parser.parse_known_args()

    run_infer(args, unknownargs)
