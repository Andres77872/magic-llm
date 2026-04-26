"""Tests for AWS SigV4 signing utility."""
import json
import pytest
from urllib.parse import quote

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest, AWSPreparedRequest
from botocore.credentials import Credentials

from magic_llm.util.sigv4 import (
    build_sigv4_headers,
    build_sigv4_prepared_request,
    build_bedrock_url,
    build_polly_url,
    resolve_credentials,
)


class TestBuildSigv4Headers:
    """Test SigV4 header generation."""

    def test_produces_authorization_header_with_aws4_hmac_sha256(self):
        """build_sigv4_headers produces valid Authorization header with AWS4-HMAC-SHA256."""
        body = json.dumps({"prompt": "test"})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"
        headers = build_sigv4_headers(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert "Authorization" in headers
        assert "AWS4-HMAC-SHA256" in headers["Authorization"]
        assert "Credential=" in headers["Authorization"]
        assert "SignedHeaders=" in headers["Authorization"]
        assert "Signature=" in headers["Authorization"]

    def test_includes_x_amz_date_header(self):
        """SigV4 headers include x-amz-date."""
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        headers = build_sigv4_headers(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert "X-Amz-Date" in headers

    def test_no_x_amz_content_sha256_for_bedrock(self):
        """SigV4 headers do NOT include x-amz-content-sha256 for Bedrock (S3-only header).

        Per AWS documentation and botocore behavior, X-Amz-Content-Sha256 is only
        required for S3 requests. Bedrock uses HTTPS transport-level integrity,
        so botocore does not add this header for bedrock-runtime requests.
        """
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        headers = build_sigv4_headers(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        # X-Amz-Content-Sha256 is S3-specific; Bedrock does not require it
        assert "X-Amz-Content-Sha256" not in headers

    def test_matches_botocore_direct_signing(self):
        """SigV4 output matches direct botocore signing for the same request."""
        body = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"
        access_key = "AKIAIOSFODNN7EXAMPLE"
        secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        region = "us-east-1"

        # Our function
        our_headers = build_sigv4_headers(
            method="POST", url=url, body=body,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            region=region, service="bedrock",
        )

        # Direct botocore signing
        creds = Credentials(access_key, secret_key)
        request = AWSRequest(method="POST", url=url, data=body)
        SigV4Auth(creds, "bedrock", region).add_auth(request)
        expected_headers = dict(request.headers)

        # Authorization headers should match (same credentials, same body, same time window)
        assert our_headers["Authorization"] == expected_headers["Authorization"]
        assert our_headers["X-Amz-Date"] == expected_headers["X-Amz-Date"]
        # X-Amz-Content-Sha256 is S3-specific; verify both sides agree (neither has it for Bedrock)
        assert ("X-Amz-Content-Sha256" in our_headers) == ("X-Amz-Content-Sha256" in expected_headers)

    def test_with_session_token(self):
        """SigV4 headers include x-amz-security-token when session_token provided."""
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        headers = build_sigv4_headers(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
            session_token="FwoGZXIvYXdzEBY...",
        )
        assert "X-Amz-Security-Token" in headers

    def test_different_service_names(self):
        """SigV4 works with different service names (bedrock, polly)."""
        body = json.dumps({"test": True})
        url = "https://polly.us-east-1.amazonaws.com/v1/speech"
        headers = build_sigv4_headers(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="polly",
        )
        assert "Authorization" in headers
        assert "AWS4-HMAC-SHA256" in headers["Authorization"]


class TestBuildBedrockUrl:
    """Test Bedrock URL construction."""

    def test_non_streaming_url(self):
        """Non-streaming URL uses /invoke endpoint."""
        url = build_bedrock_url("us-east-1", "amazon.nova-lite-v1:0", stream=False)
        assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/invoke"

    def test_streaming_url(self):
        """Streaming URL uses /invoke-with-response-stream endpoint."""
        url = build_bedrock_url("us-east-1", "amazon.nova-lite-v1:0", stream=True)
        assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/invoke-with-response-stream"

    def test_arn_model_id_uri_encoded(self):
        """ARN model IDs are URI-encoded (colons become %3A)."""
        arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"
        url = build_bedrock_url("us-east-1", arn, stream=False)
        expected_arn = quote(arn, safe='')
        assert expected_arn in url
        assert ":" not in url.split("/model/")[1].split("/")[0]  # No raw colons in model path

    def test_arn_model_id_streaming(self):
        """ARN model IDs work with streaming endpoint too."""
        arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet:1"
        url = build_bedrock_url("us-west-2", arn, stream=True)
        expected_arn = quote(arn, safe='')
        assert expected_arn in url
        assert "invoke-with-response-stream" in url

    def test_different_regions(self):
        """URL uses correct region."""
        url = build_bedrock_url("eu-west-1", "amazon.nova-lite-v1:0")
        assert "bedrock-runtime.eu-west-1.amazonaws.com" in url


class TestBuildPollyUrl:
    """Test Polly URL construction."""

    def test_polly_url(self):
        """Polly URL is correct."""
        url = build_polly_url("us-east-1")
        assert url == "https://polly.us-east-1.amazonaws.com/v1/speech"

    def test_polly_url_different_region(self):
        """Polly URL uses correct region."""
        url = build_polly_url("eu-west-1")
        assert "polly.eu-west-1.amazonaws.com" in url


class TestResolveCredentials:
    """Test credential resolution."""

    def test_explicit_credentials_returned(self):
        """Explicit credentials are returned directly."""
        result = resolve_credentials(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        assert result == ("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1", None)

    def test_partial_credentials_falls_back_to_ambient(self):
        """When only access_key is provided, falls back to ambient credentials."""
        # This will either resolve ambient creds or raise if none configured
        try:
            result = resolve_credentials(
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key=None,
                region="us-east-1",
            )
            # If ambient creds exist, access_key should be the explicit one
            assert result[0] == "AKIAIOSFODNN7EXAMPLE"
        except ValueError:
            # Expected if no ambient credentials configured
            pass

    def test_missing_credentials_raises(self):
        """When no credentials available, raises ValueError."""
        # Clear any ambient credentials for this test
        import os
        old_key = os.environ.pop("AWS_ACCESS_KEY_ID", None)
        old_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        old_session = os.environ.pop("AWS_SESSION_TOKEN", None)
        # Also clear any cached credentials file path
        old_shared = os.environ.pop("AWS_SHARED_CREDENTIALS_FILE", None)

        try:
            with pytest.raises(ValueError, match="No AWS credentials found"):
                resolve_credentials(
                    aws_access_key_id=None,
                    aws_secret_access_key=None,
                    region="us-east-1",
                )
        finally:
            # Restore environment
            if old_key is not None:
                os.environ["AWS_ACCESS_KEY_ID"] = old_key
            if old_secret is not None:
                os.environ["AWS_SECRET_ACCESS_KEY"] = old_secret
            if old_session is not None:
                os.environ["AWS_SESSION_TOKEN"] = old_session
            if old_shared is not None:
                os.environ["AWS_SHARED_CREDENTIALS_FILE"] = old_shared


class TestBuildSigv4PreparedRequest:
    """Test SigV4 prepared request generation for async transport."""

    def test_returns_aws_prepared_request(self):
        """build_sigv4_prepared_request returns an AWSPreparedRequest."""
        body = json.dumps({"prompt": "test"})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"
        prepared = build_sigv4_prepared_request(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert isinstance(prepared, AWSPreparedRequest)

    def test_includes_authorization_header(self):
        """Prepared request includes Authorization header with AWS4-HMAC-SHA256."""
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        prepared = build_sigv4_prepared_request(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert "Authorization" in prepared.headers
        assert "AWS4-HMAC-SHA256" in prepared.headers["Authorization"]

    def test_extra_headers_are_signed(self):
        """Headers passed via extra_headers are present in the signed canonical request."""
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        prepared = build_sigv4_prepared_request(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
            extra_headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        # Content-Type and Accept should be in the signed headers
        assert "Content-Type" in prepared.headers
        assert prepared.headers["Content-Type"] == "application/json"
        assert "Accept" in prepared.headers
        assert prepared.headers["Accept"] == "application/json"

    def test_url_is_preserved_with_percent_encoding(self):
        """Prepared request URL retains percent-encoded characters (e.g. %3A for colons)."""
        body = json.dumps({"test": True})
        url = build_bedrock_url("us-east-1", "amazon.nova-lite-v1:0", stream=False)
        prepared = build_sigv4_prepared_request(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        # The URL should still contain %3A (encoded colon) — not decoded
        assert "%3A" in prepared.url or "%3a" in prepared.url.lower()

    def test_body_is_bytes(self):
        """Prepared request body is always bytes."""
        prepared = build_sigv4_prepared_request(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
            body='{"test": true}',  # string body
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert isinstance(prepared.body, bytes)

    def test_body_bytes_passed_through(self):
        """Prepared request body passes bytes through unchanged."""
        raw_body = b'{"test": true}'
        prepared = build_sigv4_prepared_request(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
            body=raw_body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert prepared.body == raw_body

    def test_with_session_token(self):
        """Prepared request includes X-Amz-Security-Token when session_token provided."""
        body = json.dumps({"test": True})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke"
        prepared = build_sigv4_prepared_request(
            method="POST",
            url=url,
            body=body,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
            session_token="FwoGZXIvYXdzEBY...",
        )
        assert "X-Amz-Security-Token" in prepared.headers

    def test_matches_direct_botocore_signing(self):
        """Prepared request Authorization matches direct botocore signing with same headers."""
        body = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"
        access_key = "AKIAIOSFODNN7EXAMPLE"
        secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        region = "us-east-1"
        extra = {"Content-Type": "application/json", "Accept": "application/json"}

        # Our function
        prepared = build_sigv4_prepared_request(
            method="POST", url=url, body=body,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            region=region, service="bedrock",
            extra_headers=extra,
        )

        # Direct botocore signing with same headers
        body_bytes = body.encode('utf-8')
        creds = Credentials(access_key, secret_key)
        request = AWSRequest(method="POST", url=url, data=body_bytes, headers=extra)
        SigV4Auth(creds, "bedrock", region).add_auth(request)
        expected = request.prepare()

        assert prepared.headers["Authorization"] == expected.headers["Authorization"]
        assert prepared.headers["X-Amz-Date"] == expected.headers["X-Amz-Date"]
