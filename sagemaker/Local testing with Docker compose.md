# Launching ParMETIS partitioning jobs locally with Docker compose

This document describes how to launch Docker compose jobs that emulate a SageMaker distributed training execution environment that can be used to test distributed algorithms. We have designed this around the distributed execution of ParMETIS, but the process can be easily adapted to launch different distributed programs (e.g. DistDGL jobs).


## TLDR

1. Install Docker and docker compose: https://docs.docker.com/compose/install/linux/
2. Clone graph-storm.
3. Build the SageMaker graph-storm Docker image.
4. Generate a docker compose file:
    `python generate_sagemaker_parmetis_docker_compose.py --num-instances $NUM_INSTANCES --graph-name $GRAPH_NAME --graph-data $DATASET_S3_PATH --num-parts $NUM_PARTITIONS --output-data-s3 "s3://${OUTPUT_BUCKET}/partitioning/${DATASET_NAME}${PATH_SUFFIX}/${INSTANCE_COUNT}x-${INSTANCE_TYPE}-${NUM_PARTITIONS}parts/" --region $REGION`
5. Launch the job using docker compose: `docker compose -f "docker-compose-${GRAPH_NAME}-${NUM_INSTANCES}workers-${NUM_PARTITIONS}parts.yml" up`

## Getting Started

If you’ve never worked with Docker compose before the official description provides a great intro:

>Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services. Then, with a single command, you create and start all the services from your configuration.

We will use this capability to launch multiple worker instances locally, that will be configured to “look like” a SageMaker training instance and communicate over a virtual network created by Docker compose. This way our test environment will be as close to a real SageMaker distributed job as we can get, without needing to launch SageMaker jobs, or launch and configure multiple EC2 instances when developing features.

### Prerequisite 1: Launch or re-use a dev-ready EC2.

As we will be running multiple heavy containers is one machine we recommend using a capable Linux-based machine. We recommend at least 32GB of RAM.

### Prerequisite 2: Install Docker and Docker compose

You can follow the official Docker guides for [installation of the Docker engine](https://docs.docker.com/engine/install/).

Next you need to install the `Docker compose` plugin that will allow us to spin up multiple Docker containers. Instructions for that are [here](https://docs.docker.com/compose/install/linux/).


## Building the SageMaker graph-storm Docker image

To build the Docker image first we need to perform an ECR login
to ensure access to the source image:

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

Then, from the root of the graph-storm repo we can run the following to build and tag the image:

```
docker build -f sagemaker/docker/v1-0.0.0/Dockerfile . -t graphstorm-dev:v1-0.0.0
```

This will build the image locally which you can verify with:

```
> docker images graphstorm-dev
REPOSITORY       TAG        IMAGE ID       CREATED       SIZE
graphstorm-dev   v1-0.0.0   a42219e78261   2 hours ago   21.6GB
```

If we wanted to launch a single container based on that image and look around we can do so by running:


```
> docker run -it a42219e78261 bash
root@a2add1231c17:/opt/ml/code# # This is now a bash shell inside the container
```

OK now let’s see how we can launch multiple of these containers to run partitioning jobs locally.

## Creating the Docker compose file

A Docker compose file is a YAML file that tells Docker which containers to spin up and how to configure them. A simple example would be:


```
# file named docker-compose.yml
version: "3.9"
services:
  redis:
    image: "redis:alpine"
```

The above docker compose file launches a single service, `redis` that is an instance of the image `redis:alpine`.

To launch the services described in the above file we can use `docker compose -f docker-compose.yml up`. This will launch the container and execute its entry point.

To emulate a SageMaker distributed execution environment based on the image we built previously we would need a Docker compose file that looks like this:

```
version: '3.7'

networks:
  mpi:
    name: mpi-network

services:
  algo-1:
    image: graphstorm-dev:v1-0.0.0
    container_name: algo-1
    hostname: algo-1
    networks:
      - mpi
    command: 'python3 sagemaker_parmetis.py --graph-name "ml-25M-imdb-edge-class" --graph-data-s3 "s3://my-bucket/ml-25M-imdb-edge-class/" --num-parts 4 --output-data-s3 "s3://my-bucket/output/partitioning/ml-25M-docker-test/"'
    environment:
      SM_TRAINING_ENV: '{"hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "current_host": "algo-1"}'
      WORLD_SIZE: 4
      MASTER_ADDR: 'algo-1'
      AWS_REGION: 'us-west-2'
    ports:
      - 22
    working_dir: '/opt/ml/code/'

  algo-2:
      [...]
```

Some explanation on the above elements (see the [official docs](https://docs.docker.com/compose/compose-file/) for more details):

* `image`: Determines which image we’ll use for the container launched.
* `environment`: Determines the environment variables that will be set for the container once it launches.
* `command`: Determines the entrypoint, i.e. the command that will be executed once the container launches.

Instead of creating this file by hand every time we want to modify the number of instances or the dataset we’ll use, we have created a Python script that builds the docker compose file for us, `generate_sagemaker_parmetis_docker_compose.py`. Note that the script uses the [PyYAML](https://pypi.org/project/PyYAML/) library.

This file has 6 required arguments that determine the Docker compose file that will be generated:

* `--num-instances`: The number of instances we want to launch. This will determine the number of `algo-x` `services` entries our compose file ends up with.
* The rest of the arguments are passed on to `sagemaker_parmetis.py`
* `--num-parts`: The number of partitions requested. Usually this should equal `num-instances`.
* `--graph-name`: The name of the graph.
* `--graph-data-s3`: The S3 URI path to the input data.
* `--output-data-s3`: The S3 URI path to where the output will be created.
* `--region`: The region in which execution will take place.


The above will create a Docker compose file named `docker-compose-${graph_name}-${num-instances}workers-${num-parts}parts.yml`, which we can then use to launch the job with (for example):


```
docker compose -f docker-compose-ml-25M-imdb-edge-class-4workers-4parts.yml up
```

Running the above command will launch 4 instances of the image, configured with the command and env vars that emulate a SageMaker execution environment and run the `sagemaker_parmetis.py` code. Note that the containers actually
interact with S3 so you would require valid AWS credentials to run.

To combine the steps we can use a simple bash script:

```
# run_parmetis.sh
# Rebuilds the image, creates a docker compose file and launches a ParMETIS job
# in a local cluster. Assumes execution from the root of the graph-storm repository
DATASET_S3_PATH="s3://my-bucket/ml-25M-imdb-edge-class/"
OUTPUT_BUCKET="my-output-bucket"
DATASET_NAME="ml-25M-imdb-edge-class"
GRAPH_NAME="ml-25M-imdb-edge-class"
METADATA_FILE="metadata.json"
INSTANCE_COUNT="8"
INSTANCE_TYPE="ml.m5.4xlarge"
NUM_PARTITIONS="8"
NUM_INSTANCES="8"
REGION="us-west-2"
IMAGE_VERSION="v1-0.0.0"
PATH_SUFFIX=""

# Rebuild the image
docker build -f sagemaker/docker/v1-0.0.0/Dockerfile . -t graphstorm-dev:v1-0.0.0

# Create the docker compose file
python sagemaker/generate_sagemaker_parmetis_docker_compose.py --num-instances $NUM_INSTANCES \
    --graph-name $GRAPH_NAME --graph-data $DATASET_S3_PATH --num-parts $NUM_PARTITIONS \
    --output-data-s3 "s3://${OUTPUT_BUCKET}/partitioning/${DATASET_NAME}${PATH_SUFFIX}/${INSTANCE_COUNT}x-${INSTANCE_TYPE}-${NUM_PARTITIONS}parts/" \
    --region $REGION

# Launch the containers
docker compose -f "docker-compose-${GRAPH_NAME}-${NUM_INSTANCES}workers-${NUM_PARTITIONS}parts.yml" up
```


And that’s it! With the above setup you’ll be able to make local modifications to graphstorm, re-build the graphstorm image and immediately launch a job that tests those changes, for any dataset/instance count you want (as long as your instance doesn’t run out of memory), as if they were being executed in SageMaker*. This setup is also faster and easier than having to launch and maintain a cluster of EC2 instances for your testing.

** There will be differences in the execution context of course, so you’ll still need integration tests to ensure full compatibility.*
