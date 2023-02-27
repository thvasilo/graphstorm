# !/bin/bash
date

# process argument 1: graphstorm home folder 
if [ -z "$1" ]; then
    echo "Please provide the graphstorm home folder that the graphstorm codes are cloned to."
    echo "For example, bash ./build_pip.sh /graph-storm/"
    exit 1
else
    GSF_HOME="$1"
fi

# process argument 2: graphstorm S3 bucket path 
if [ -z "$2" ]; then
    S3_PATH="s3://graphstorm-artifacts/"
else
    S3_PATH="$2"
fi

# The build script requires python3 installed, otherwise all below commands will fail
python3 -m pip install hatchling

cd $GSF_HOME

COMMITHASH=`git rev-parse HEAD`
SHORTCOMMIT=${COMMITHASH:0:8}
echo "The current commit hash: ${SHORTCOMMIT}"
echo ''

# replace the release version with current hash value's first 8 letters/digits
cat ./pyproject.toml > /tmp/pyproject.toml && \
sed "s/0.0.1+[0-9a-z]*/0.0.1+${SHORTCOMMIT}/" /tmp/pyproject.toml > ./pyproject.toml

# build the pip .whl file
python3 -m build

# push the .whl file into repository
# need to configure AKSK for ECR operation
WHEEL_FILE="graphstorm-0.0.1+${SHORTCOMMIT}-py3-none-any.whl"
aws s3 cp $GSF_HOME"dist/"$WHEEL_FILE $S3_PATH""$WHEEL_FILE
