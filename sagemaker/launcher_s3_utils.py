"""
Utility functions for working with S3.
"""
import logging
from typing import List

import boto3

def list_s3_objects(bucket: str, prefix: str, s3_boto_client=None) -> List[str]:
    """
    Lists all objects under provided S3 bucket and prefix. Returns ordered list of object key paths.

    Note that the function returns key paths, not the full S3 uri.
    E.g. if called with list_s3_objects('my-bucket', 'my-prefix/'), this function will return:
    [
        'my-prefix/file1.txt',
        'my-prefix/file2.txt',
        'my-prefix/subdir/file3.txt',
    ]
    """
    s3_boto_client = boto3.client('s3') if s3_boto_client is None else s3_boto_client
    paginator = s3_boto_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    object_list = []
    for page in pages:
        for obj in page['Contents']:
            object_list.append(obj['Key'])

    return sorted(object_list)

def determine_byte_size_on_s3(bucket: str, prefix: str, s3_boto_client=None) -> List[str]:
    """
    Returns the total byte size under all files under a common prefix.
    """
    if prefix.startswith("s3://"):
        logging.warning(
            "Key %s looks like an S3 URI, stripping its prefix to get data size.",
            prefix)
        prefix = prefix.replace(f"s3://{bucket}", "")
    s3_boto_client = boto3.client('s3') if s3_boto_client is None else s3_boto_client
    paginator = s3_boto_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    total_object_size_in_bytes = 0
    for page in pages:
        for obj in page['Contents']:
            total_object_size_in_bytes += int(obj['Size'])

    return total_object_size_in_bytes
