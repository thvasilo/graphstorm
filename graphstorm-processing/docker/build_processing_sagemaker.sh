#!/bin/bash
set -eox pipefail

# process argument 1: graphstorm home folder
if [ -z "$1" ]; then
    echo "Please provide the graphstorm-processing base directory"
    echo "For example, ./graphstorm-processing/docker/build_processing_sagemaker.sh /path/to/graphstorm/graphstorm-processing/"
    exit 1
else
    GSP_HOME="$1"
fi

VERSION=`poetry version --short`

# process argument 2: docker image name, default is graphstorm-processing
if [ -z "$2" ]; then
    IMAGE_NAME="graphstorm-processing"
else
    IMAGE_NAME="$2"
fi

# process argument 3: image's tag name, default is
if [ -z "$3" ]; then
    TAG=${VERSION}
else
    TAG="$3"
fi

# Prepare Docker build directory
rm -rf "${GSP_HOME}/docker/code"
mkdir -p "${GSP_HOME}/docker/code"

# Build the graphstorm-processing library
poetry build -C ${GSP_HOME} --format wheel

# Copy required files to the Docker build folder
cp -r ${GSP_HOME}/graphstorm_processing/ "${GSP_HOME}/docker/code/"
cp ${GSP_HOME}/dist/graphstorm_processing-${VERSION}-py3-none-any.whl \
    "${GSP_HOME}/docker/code"

DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}"

# Login to ECR for source SageMaker image
aws ecr get-login-password --region us-west-2 \
    | docker login --username AWS --password-stdin 153931337802.dkr.ecr.us-west-2.amazonaws.com

echo "Build a SageMaker docker image ${DOCKER_FULLNAME}"
DOCKER_BUILDKIT=1 docker build -f "${GSP_HOME}/docker/${VERSION}/Dockerfile.cpu" \
    "${GSP_HOME}/docker/" -t $DOCKER_FULLNAME

# remove the temporary code folder
rm -rf "${GSP_HOME}/docker/code"
