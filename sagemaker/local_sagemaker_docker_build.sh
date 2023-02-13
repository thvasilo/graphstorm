#!/usr/bin/env bash

set -e
# Check to see if input has been provided:
if [ -z "$1" ]; then
    echo "Please provide the S3 bucket name to deploy the build."
    echo "For example: ./local_sagemaker_docker_build.sh my-bucket"
    echo "You can also provide the region for example: ./local_sagemaker_docker_build.sh my-bucket us-west-2 v1-0.0.0"
    exit 1
fi

base_dir="$PWD"
echo $base_dir

# Get the region defined in the current configuration (default to us-west-2 if none defined)
if [ -z "$2" ]; then
    region=$(aws configure get region)
    region=${region:-us-west-2}
else
    region="$2"
fi

latest_version="v1-0.0.0"
if [ -z "$3" ]; then
   version=$latest_version
else
   version=$3
fi

# build and package to s3
prefix='graphstorm'
s3_prefix="s3://$1/${prefix}"

rm -rf code
mkdir -p code
mkdir -p code/graphstorm/

# copy the required folders the distribution folder
cp -r python code/graphstorm/
cp -r tests code/graphstorm/
cp -r sagemaker code/graphstorm/
cp -r tools code/graphstorm/
cp -r training_scripts code/graphstorm/
cp -r inference_scripts code/graphstorm/

aws s3 rm --recursive --quiet $s3_prefix/code/
aws s3 sync --quiet code/ $s3_prefix/code/

account=$(aws sts get-caller-identity --query Account --output text)


bash sagemaker/docker/local_build_and_push.sh graphstorm-dev ${version} ${region} ${account}
