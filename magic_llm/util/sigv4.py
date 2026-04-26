"""
AWS Signature Version 4 (SigV4) signing utility.

Uses botocore for signing only — no boto3/aioboto3 SDK clients.
This module provides:
- build_sigv4_headers(): Sign HTTP requests with AWS SigV4
- build_sigv4_prepared_request(): Sign and prepare request for async transport (aiohttp/yarl-safe)
- build_bedrock_url(): Construct Bedrock REST endpoint URLs
- build_polly_url(): Construct Polly REST endpoint URLs
- resolve_credentials(): Resolve AWS credentials (explicit or ambient IAM chain)
"""

import asyncio
from typing import Optional
from urllib.parse import quote

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest, AWSPreparedRequest
from botocore.credentials import Credentials
from botocore.session import Session


def build_bedrock_url(region: str, model_id: str, stream: bool = False) -> str:
    """
    Construct a Bedrock REST endpoint URL.

    Args:
        region: AWS region (e.g. 'us-east-1')
        model_id: Model identifier (e.g. 'amazon.nova-lite-v1:0' or ARN).
            ARN model IDs are URI-encoded to ensure valid URL paths.
        stream: If True, use the streaming endpoint

    Returns:
        Full Bedrock URL for the model
    """
    endpoint = 'invoke-with-response-stream' if stream else 'invoke'
    # URI-encode model_id to handle ARNs containing colons and other special chars
    encoded_model_id = quote(model_id, safe='')
    return f'https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model_id}/{endpoint}'


def build_polly_url(region: str) -> str:
    """
    Construct a Polly REST endpoint URL.

    Args:
        region: AWS region (e.g. 'us-east-1')

    Returns:
        Full Polly URL for speech synthesis
    """
    return f'https://polly.{region}.amazonaws.com/v1/speech'


def build_sigv4_headers(
    method: str,
    url: str,
    body: str | bytes,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region: str,
    service: str = 'bedrock',
    session_token: str | None = None,
) -> dict[str, str]:
    """
    Build AWS SigV4-signed headers for a raw HTTP request.

    Args:
        method: HTTP method (POST, GET, etc.)
        url: Full URL (e.g. https://bedrock-runtime.us-east-1.amazonaws.com/model/...)
        body: Request body as string or bytes (JSON for Bedrock)
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret key
        region: AWS region (e.g. 'us-east-1')
        service: AWS service name (default: 'bedrock' — NOT 'bedrock-runtime';
            the signing name is 'bedrock' while the endpoint prefix is 'bedrock-runtime')
        session_token: Optional session token for temporary IAM credentials

    Returns:
        Dict of headers to include in the HTTP request (Authorization, x-amz-date, etc.)
    """
    creds = Credentials(aws_access_key_id, aws_secret_access_key, token=session_token)
    request = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(creds, service, region).add_auth(request)
    return dict(request.headers)


def build_sigv4_prepared_request(
    method: str,
    url: str,
    body: str | bytes,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region: str,
    service: str = 'bedrock',
    session_token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AWSPreparedRequest:
    """
    Sign an HTTP request with AWS SigV4 and return an AWSPreparedRequest.

    Stable headers (Content-Type, Accept, etc.) MUST be passed in extra_headers
    so they are present at signing time and included in the canonical request.

    The returned AWSPreparedRequest provides:
    - ``.url``: The fully-encoded URL string (safe for yarl.URL(..., encoded=True))
    - ``.headers``: A CaseInsensitiveDict with all signed + transport headers
    - ``.body``: The finalized request body as bytes

    This is the correct handoff pattern for aiohttp transport, matching what
    aiobotocore does internally to avoid yarl URL re-encoding issues.

    Args:
        method: HTTP method (POST, GET, etc.)
        url: Full URL (e.g. https://bedrock-runtime.us-east-1.amazonaws.com/model/...)
        body: Request body as string or bytes (JSON for Bedrock)
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret key
        region: AWS region (e.g. 'us-east-1')
        service: AWS service name (default: 'bedrock')
        session_token: Optional session token for temporary IAM credentials
        extra_headers: Headers that MUST be present before signing (e.g. Content-Type, Accept)

    Returns:
        AWSPreparedRequest ready for aiohttp transport
    """
    body_bytes = body.encode('utf-8') if isinstance(body, str) else (body or b'')
    headers = dict(extra_headers) if extra_headers else {}

    request = AWSRequest(
        method=method,
        url=url,
        data=body_bytes,
        headers=headers,
    )
    creds = Credentials(aws_access_key_id, aws_secret_access_key, token=session_token)
    SigV4Auth(creds, service, region).add_auth(request)
    return request.prepare()


def resolve_credentials(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region: str = 'us-east-1',
) -> tuple[str, str, str, Optional[str]]:
    """
    Resolve AWS credentials for SigV4 signing.

    If explicit credentials are provided, returns them directly.
    Otherwise, uses botocore's credential chain (environment variables,
    shared credentials file, IAM role, etc.) via asyncio.to_thread() for
    async-safe resolution.

    Args:
        aws_access_key_id: Explicit AWS access key (optional)
        aws_secret_access_key: Explicit AWS secret key (optional)
        region: AWS region (default: 'us-east-1')

    Returns:
        Tuple of (aws_access_key_id, aws_secret_access_key, region, session_token)
    """
    if aws_access_key_id and aws_secret_access_key:
        return aws_access_key_id, aws_secret_access_key, region, None

    # Use botocore's ambient credential chain (env vars, ~/.aws/credentials, IAM role)
    session = Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError(
            "No AWS credentials found. Provide aws_access_key_id and "
            "aws_secret_access_key, or configure ambient credentials "
            "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars, ~/.aws/credentials, etc.)"
        )

    frozen = credentials.get_frozen_credentials()
    resolved_access_key = frozen.access_key or aws_access_key_id
    resolved_secret_key = frozen.secret_key or aws_secret_access_key

    if not resolved_access_key or not resolved_secret_key:
        raise ValueError("Resolved AWS credentials are incomplete (missing access_key or secret_key)")

    return resolved_access_key, resolved_secret_key, region, frozen.token


async def resolve_credentials_async(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region: str = 'us-east-1',
) -> tuple[str, str, str, Optional[str]]:
    """
    Async wrapper for resolve_credentials using asyncio.to_thread().

    Freezes ambient credentials off the event loop to avoid blocking.

    Args:
        aws_access_key_id: Explicit AWS access key (optional)
        aws_secret_access_key: Explicit AWS secret key (optional)
        region: AWS region (default: 'us-east-1')

    Returns:
        Tuple of (aws_access_key_id, aws_secret_access_key, region, session_token)
    """
    if aws_access_key_id and aws_secret_access_key:
        return aws_access_key_id, aws_secret_access_key, region, None

    return await asyncio.to_thread(
        resolve_credentials,
        aws_access_key_id,
        aws_secret_access_key,
        region,
    )
