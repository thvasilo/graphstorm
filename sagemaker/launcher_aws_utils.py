"""
This module contains utility functions for AWS resources.
"""
from typing import Optional

import boto3
from botocore.config import Config

def get_quota_value(service_code: str, quota_name: str, region: str) -> str:
    """
    For a give service code, quota name and region return the quota value.
    """
    def get_value_for_quota_name(quotas: str, quota_name: str):
        for quota in quotas:
            if quota['QuotaName'] == quota_name:
                return quota['Value'], quota['QuotaCode']
        return None, None

    quota_client = boto3.client(
        'service-quotas',
        config=Config(
            region_name=region, connect_timeout=5, read_timeout=60, retries={'max_attempts': 20}))
    quota_paginator_client = quota_client.get_paginator('list_service_quotas')

    quota_response = quota_paginator_client.paginate(
        ServiceCode=service_code,
    )

    for quota_set in quota_response:
        quota_value, _ = get_value_for_quota_name(quota_set['Quotas'], quota_name)
        if quota_value is not None:
            break

    return quota_value

def check_if_instances_available(
        task: str, instance_type: str, instance_count: int, region: str) -> None:
    """
    Checks if we have enough of a quota for the requested instance type, task and count in a region.
    Exits the program if not.
    """
    if task not in {'training', 'processing', 'transform', 'spot training'}:
        raise RuntimeError(
            f"SageMaker instance count can only be"
            f" determined for training/processing/transform instances, {task} requested")
    quota_name = f"{instance_type} for {task} job usage"
    quota_value = get_quota_value('sagemaker', quota_name, region)

    if quota_value is None:
        raise RuntimeError(
            f"Cannot find quota for {task} using instance type"
            f" {instance_type}. Check if the instance type and region are correct")
    if int(quota_value) < instance_count:
        raise RuntimeError(
            f"Requested instance count {instance_count}"
            f" goes above current quota ({quota_value}) for instance type {instance_type}")

    # TODO: Use sagemaker_client.list_{task}_jobs and describe_{task}_job to get current utilization

def get_max_volume_size_for_sagemaker(task: str, region: str) -> Optional[int]:
    """
    Get the maximum allowed EBS volume size for a [training/processing/transform]
    instance for the region.
    Returns an integer value, or None if it's not possible to retrieve the quota.
    """
    if task not in {'training', 'processing', 'transform', 'spot training'}:
        raise RuntimeError(
            f"SageMaker volume size can only be"
            f" determined for training/processing/transform instances, {task} requested")
    quota_name = f"Size of EBS volume for a {task} job instance"
    quota_value = get_quota_value('sagemaker', quota_name, region)
    if quota_value:
        return int(quota_value)
    else:
        return None
