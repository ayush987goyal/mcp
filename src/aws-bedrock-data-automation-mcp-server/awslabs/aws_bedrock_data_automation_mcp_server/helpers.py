"""Helper functions for the AWS Bedrock Data Automation MCP Server."""

import asyncio
import boto3
import json
import os
import uuid
from loguru import logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def get_region() -> str:
    """Get the AWS region from environment variables."""
    return os.environ.get('AWS_REGION', 'us-east-1')


def get_account_id() -> Optional[str]:
    """Get the AWS account ID from environment variables."""
    account_id = os.environ.get('AWS_ACCOUNT_ID')
    if not account_id:
        raise ValueError('AWS_ACCOUNT_ID environment variable is not set')
    return account_id


def get_bucket_name() -> Optional[str]:
    """Get the S3 bucket name from environment variables."""
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError('AWS_BUCKET_NAME environment variable is not set')
    return bucket_name


def get_aws_session(region_name=None):
    """Create an AWS session using AWS Profile or default credentials."""
    profile_name = os.environ.get('AWS_PROFILE')
    region = region_name or get_region()

    if profile_name:
        logger.debug(f'Using AWS profile: {profile_name}')
        return boto3.Session(profile_name=profile_name, region_name=region)
    else:
        logger.debug('Using default AWS credential chain')
        return boto3.Session(region_name=region)


def get_profile_arn() -> Optional[str]:
    """Get the Bedrock Data Automation profile ARN."""
    region = get_region()
    account_id = get_account_id()
    return f'arn:aws:bedrock:{region}:{account_id}:data-automation-profile/us.data-automation-v1'


def get_bedrock_data_automation_client():
    """Get a Bedrock Data Automation client."""
    session = get_aws_session()
    return session.client('bedrock-data-automation', region_name=get_region())


def get_bedrock_data_automation_runtime_client():
    """Get a Bedrock Data Automation Runtime client."""
    session = get_aws_session()
    return session.client('bedrock-data-automation-runtime', region_name=get_region())


def get_s3_client():
    """Get an S3 client."""
    session = get_aws_session()
    return session.client('s3', region_name=get_region())


async def list_projects() -> list:
    """List all Bedrock Data Automation projects."""
    client = get_bedrock_data_automation_client()
    response = client.list_data_automation_projects()
    return response.get('projects', [])


async def get_project(project_arn: str) -> Dict[str, Any]:
    """Get details of a Bedrock Data Automation project.

    Args:
        project_arn: The ARN of the project to get details for.

    Returns:
        The project details.
    """
    client = get_bedrock_data_automation_client()
    response = client.get_data_automation_project(projectArn=project_arn)
    return response.get('project', {})


async def upload_to_s3(asset_path: str) -> str:
    """Upload an asset to S3.

    Args:
        asset_path: The path to the asset to upload.

    Returns:
        The S3 URI of the uploaded asset.

    Raises:
        ValueError: If the bucket name is not set or the asset does not exist.
    """
    bucket_name = get_bucket_name()
    asset_path_obj = Path(asset_path)
    if not asset_path_obj.exists():
        raise ValueError(f'Asset at path {asset_path} does not exist')

    with open(asset_path, 'rb') as f:
        asset_content = f.read()

    extension = asset_path_obj.suffix
    key = f'mcp/{uuid.uuid4()}{extension}'

    s3_client = get_s3_client()
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=asset_content)

    logger.info(f'Uploaded {asset_path} to s3://{bucket_name}/{key}')
    return f's3://{bucket_name}/{key}'


def get_bucket_and_key_from_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse an S3 URI into bucket and key.

    Args:
        s3_uri: The S3 URI to parse.

    Returns:
        A tuple of (bucket, key).
    """
    parts = s3_uri.split('/')
    bucket = parts[2]
    key = '/'.join(parts[3:])
    return bucket, key


async def download_from_s3(s3_uri: str) -> Optional[Dict[str, Any]]:
    """Download and parse a JSON file from S3.

    Args:
        s3_uri: The S3 URI to download from.

    Returns:
        The parsed JSON content, or None if the download fails.
    """
    bucket, key = get_bucket_and_key_from_s3_uri(s3_uri)

    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        raise ValueError(f'Error downloading from S3: {e}')


async def invoke_data_automation_and_get_results(
    asset_path: str, project_arn: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Invoke a Bedrock Data Automation job and get the results.

    Args:
        asset_path: The path to the asset to process.
        project_arn: The ARN of the project to use. If not provided, uses the default public project.

    Returns:
        The job results, or None if the job fails.

    Raises:
        ValueError: If the profile ARN is not available.
    """
    asset_uri = await upload_to_s3(asset_path)

    if not project_arn:
        region = get_region()
        project_arn = f'arn:aws:bedrock:{region}:aws:data-automation-project/public-default'

    logger.info(f'Using assetUri: {asset_uri} and projectArn: {project_arn}')

    profile_arn = get_profile_arn()
    bucket_name = get_bucket_name()
    runtime_client = get_bedrock_data_automation_runtime_client()

    # Invoke the data automation job
    response = runtime_client.invoke_data_automation_async(
        inputConfiguration={'s3Uri': asset_uri},
        outputConfiguration={'s3Uri': f's3://{bucket_name}/mcp/test-output'},
        dataAutomationConfiguration={'dataAutomationProjectArn': project_arn},
        dataAutomationProfileArn=profile_arn,
    )

    invocation_arn = response.get('invocationArn')
    logger.info(f'Data Automation invoked: {invocation_arn}')

    # Poll for job completion
    while True:
        get_response = runtime_client.get_data_automation_status(invocationArn=invocation_arn)
        status = get_response.get('status')

        if status != 'InProgress':
            break

        # Wait before polling again
        await asyncio.sleep(3)

    logger.info(f'Data Automation completed: {get_response}')

    if status != 'Success' or not get_response.get('outputConfiguration', {}).get('s3Uri'):
        raise ValueError(f'Data Automation failed: {get_response}')

    output_uri = get_response['outputConfiguration']['s3Uri']
    job_metadata = await download_from_s3(output_uri)

    logger.info(f'Job metadata: {job_metadata}')

    # Extract output paths
    standard_output_uri = None
    custom_output_uri = None

    if job_metadata is not None:
        try:
            standard_output_uri = job_metadata['output_metadata'][0]['segment_metadata'][0].get(
                'standard_output_path'
            )
        except (KeyError, IndexError):
            standard_output_uri = None

        try:
            custom_output_uri = job_metadata['output_metadata'][0]['segment_metadata'][0].get(
                'custom_output_path'
            )
        except (KeyError, IndexError):
            custom_output_uri = None

    if not standard_output_uri and not custom_output_uri:
        raise ValueError('Data Automation failed. No standard or custom output found')

    result: Dict[str, Optional[Dict[str, Any]]] = {'standardOutput': None, 'customOutput': None}

    if standard_output_uri:
        standard_output = await download_from_s3(standard_output_uri)
        if standard_output is not None:
            result['standardOutput'] = standard_output

    if custom_output_uri:
        custom_output = await download_from_s3(custom_output_uri)
        if custom_output is not None:
            result['customOutput'] = custom_output

    return result
