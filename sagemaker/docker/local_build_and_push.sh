#!/usr/bin/env bash

set -euo pipefail
# Usage help
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: docker/local_build_and_push.sh"
    echo "Optionally provide the image name, tag, region and account number for the ecr repository"
    echo "For example: docker/local_build_and_push.sh graphstorm-dev v1-0.0.0 us-west-2 1234567890"
    exit 1
fi

latest_version="v1-0.0.0" # needs to be updated anytime there's a new version

# TODO: Use proper flags for these arguments instead of relying on position
# Set the image name/repository
if [ -b "${1-}" ]; then
    image="$1"
else
    image='graphstorm-dev'
fi

# Set the image tag/version
if [ -n "${2-}" ]; then
    version="$2"
else
    version=${latest_version}
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
if [ -n "${3-}" ]; then
    region="$3"
else
    region=$(aws configure get region)
    region=${region:-us-west-2}
fi

# Get the account number associated with the current IAM credentials
if [ -n "${4-}" ]; then
    account=$4
else
    account=$(aws sts get-caller-identity --query Account --output text)
fi

suffix="${version}"
latest_suffix="latest"

echo "ecr image: ${image},
version: ${version},
region: ${region},
account: ${account}"

# Prepare container folder
rm -rf container
mkdir -p container

dgl_package="../dgl"
graphstorm_package='.'
source_code_prefix="$graphstorm_package/code"

source_dockerfile="sagemaker/docker/${version}/Dockerfile"

echo "Copying source code files from ${source_code_prefix} and dockerfile from ${source_dockerfile} to container directory"
cp -r $source_code_prefix container
rsync -r $dgl_package container --exclude .git --exclude third_party
cp $source_dockerfile container/Dockerfile

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${suffix}"
latest_tag="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${latest_suffix}"

# If the repository doesn't exist in ECR, create it.
echo "Getting or creating container repository: ${image}"
if ! $(aws ecr describe-repositories --repository-names "${image}" --region ${region} > /dev/null 2>&1); then
    echo "Container repository ${image} does not exist. Creating"
    aws ecr create-repository --repository-name "${image}" --region ${region} > /dev/null
fi

echo "Get the docker login command from ECR and execute it directly"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR with the full name.
echo "Building container image"

echo "Pull the docker image from ${fullname} to use remote cache"
docker pull ${fullname} || true

docker build --cache-from ${fullname} -t ${image}:${suffix} container
docker tag ${image}:${suffix} ${fullname}

docker push ${fullname}

if [ ${version} = ${latest_version} ]; then
    docker tag ${image}:${suffix} ${latest_tag}
    docker push ${latest_tag}
fi