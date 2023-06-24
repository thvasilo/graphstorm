#!/usr/bin/env bash

set -euo pipefail
# Usage help
if [ -b "${1-}" ] && [ "$1" == "--help" ] || [ -b "${1-}" ] && [ "$1" == "-h" ]; then
    echo "Usage: docker/push_sm_container.sh <image-name>"
    echo "Optionally provide the image name, tag, region and account number for the ecr repository"
    echo "For example: docker/push_sm_container.sh graphstorm-sm us-west-2 1234567890"
    exit 1
fi

version=`git rev-parse --short HEAD`

# TODO: Use proper flags for these arguments instead of relying on position
# Set the image name/repository
if [ -b "${1-}" ]; then
    image="$1"
else
    image='graphstorm-sm'
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
if [ -n "${2-}" ]; then
    region="$2"
else
    region=$(aws configure get region)
    region=${region:-us-west-2}
fi

# Get the account number associated with the current IAM credentials
if [ -n "${3-}" ]; then
    account=$3
else
    account=$(aws sts get-caller-identity --query Account --output text)
fi

suffix="${version}"
latest_suffix="latest"

echo "ecr image: ${image},
version: ${version},
region: ${region},
account: ${account}"

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

# Build the docker image locally with the image name and then push it to ECR with the full name.
echo "Building container image"

docker tag ${image}:${suffix} ${fullname}

docker push ${fullname}

docker tag ${image}:${suffix} ${latest_tag}
docker push ${latest_tag}
