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

    Launch SageMaker training task
"""
import os

from sagemaker.pytorch.estimator import PyTorch

from common_parser import (
    create_sm_session,
    get_common_parser,
    parse_estimator_kwargs,
    SUPPORTED_TASKS,
    parse_unknown_gs_args,
    )

INSTANCE_TYPE = "ml.g4dn.12xlarge"

def run_job(input_args, image, unknowargs):
    """ Run job using SageMaker estimator.PyTorch

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    unknowargs: dict
        GraphStorm parameters
    """
    sm_task_name = input_args.task_name # SageMaker task name
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    instance_count = input_args.instance_count # Number of infernece instances
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm training entry_point
    task_type = input_args.task_type # Training task type
    graph_name = input_args.graph_name # Training graph name
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing partitioned graph data
    train_yaml_s3 = input_args.yaml_s3 # S3 location storing the yaml file
    model_artifact_s3 = input_args.model_artifact_s3 # Where to store model artifacts
    model_checkpoint_to_load = input_args.model_checkpoint_to_load # S3 location of a saved model.
    custom_script = input_args.custom_script # custom_script if any
    log_level = input_args.log_level # SageMaker runner logging level

    sess = create_sm_session(instance_type, region)

    container_image_uri = image

    params = {
        "graph-data-s3": graph_data_s3,
        "graph-name": graph_name,
        "log-level": log_level,
        "model-artifact-s3": model_artifact_s3,
        "task-type": task_type,
        "train-yaml-s3": train_yaml_s3,
    }
    if custom_script is not None:
        params["custom-script"] = custom_script
    if model_checkpoint_to_load is not None:
        params["model-checkpoint-to-load"] = model_checkpoint_to_load
    unknown_args_dict = parse_unknown_gs_args(unknowargs)
    params.update(unknown_args_dict)

    print(f"SageMaker launch parameters {params}")
    print(f"GraphStorm forwarded parameters {unknown_args_dict}")
    if input_args.sm_estimator_parameters:
        print(f"SageMaker Estimator parameters: '{input_args.sm_estimator_parameters}'")

    estimator_kwargs = parse_estimator_kwargs(input_args.sm_estimator_parameters)

    est = PyTorch(
        entry_point=os.path.basename(entry_point),
        hyperparameters=params,
        image_uri=container_image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=model_artifact_s3,
        py_version="py3",
        role=role,
        sagemaker_session=sess,
        source_dir=os.path.dirname(entry_point),
        tags=[{"Key":"GraphStorm", "Value":"oss"},
              {"Key":"GraphStorm_Task", "Value":"Training"}],
        **estimator_kwargs
    )

    est.fit({"train": train_yaml_s3}, job_name=sm_task_name, wait=not input_args.async_execution)

def get_train_parser():
    """
    Return a parser for a GraphStorm training task.
    """
    parser = get_common_parser()

    training_args = parser.add_argument_group("Training Arguments")

    training_args.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/train_entry.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_train.py")

    # task specific
    training_args.add_argument("--graph-name", type=str, help="Graph name",
        required=True)
    training_args.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}", required=True)
    training_args.add_argument("--yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph", required=True)
    training_args.add_argument("--model-artifact-s3", type=str, default=None,
        help="S3 path to save model artifacts")
    training_args.add_argument("--model-checkpoint-to-load", type=str, default=None,
        help="S3 path to a model checkpoint from a previous training task "
             "that is going to be resumed.")
    training_args.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")
    training_args.add_argument('--log-level', default='INFO',
        type=str, choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])

    return parser

if __name__ == "__main__":
    arg_parser = get_train_parser()
    args, unknownargs = arg_parser.parse_known_args()
    print(f"Train launch Known args: '{args}'")
    print(f"Train launch unknown args:{type(unknownargs)=} '{unknownargs=}'")

    train_image = args.image_url

    if not args.instance_type:
        args.instance_type = INSTANCE_TYPE

    run_job(args, train_image, unknownargs)
