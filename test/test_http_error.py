"""Tests for HttpError exception class (magic_llm.util.http)."""

import pytest

from magic_llm.util.http import HttpError
from magic_llm.util.provider_errors import MESSAGES, provider_error_detail


class TestHttpError:
    """HttpError is a simple exception with optional status_code and response_content."""

    def test_message_only(self):
        err = HttpError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.status_code is None
        assert err.response_content is None

    def test_with_status_code(self):
        err = HttpError("not found", status_code=404)
        assert str(err) == "not found"
        assert err.status_code == 404
        assert err.response_content is None

    def test_with_response_content(self):
        """A body is classified and dropped; the text is the safe message."""
        body = b'{"error": "not found"}'
        err = HttpError("not found", status_code=404, response_content=body)
        assert str(err) == MESSAGES["PROVIDER_UNAVAILABLE"] + " (HTTP 404)"
        assert err.status_code == 404
        assert err.response_content is None

    def test_response_content_only(self):
        body = b'bad response'
        err = HttpError("bad", response_content=body)
        assert err.status_code is None
        assert err.response_content is None
        assert str(err) == MESSAGES["PROVIDER_UNAVAILABLE"]

    def test_is_exception(self):
        err = HttpError("test")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(HttpError, match="boom"):
            raise HttpError("boom", status_code=500)

    def test_args_attribute_accessible(self):
        err = HttpError("err", status_code=503)
        assert err.args == ("err",)


class TestHttpErrorClassification:
    """What survives a body: the typed error kind and the classification."""

    def test_plain_400_naming_the_context_window_is_a_context_overflow(self):
        body = b'{"error": {"message": "This model\'s maximum context length is 8192 tokens.", "type": "invalid_request_error"}}'
        err = HttpError("HTTP 400", 400, body)
        assert err.code == "PROVIDER_CONTEXT_EXCEEDED"
        assert err.error_type == "invalid_request_error"
        assert "8192" not in str(err) + repr(vars(err))

    def test_openrouter_metadata_error_type_wins(self):
        body = b'{"error": {"code": 400, "message": "too big", "metadata": {"error_type": "context_length_exceeded"}}}'
        err = HttpError("HTTP 400", 400, body)
        assert err.error_type == "context_length_exceeded"
        assert err.code == "PROVIDER_CONTEXT_EXCEEDED"

    def test_413_is_a_context_overflow(self):
        assert HttpError("HTTP 413", 413).code == "PROVIDER_CONTEXT_EXCEEDED"

    def test_402_is_quota(self):
        assert HttpError("HTTP 402", 402).code == "PROVIDER_QUOTA_EXCEEDED"

    def test_explicit_error_type_is_kept_only_when_recognized(self):
        assert HttpError("x", 400, error_type="token_limit_exceeded").error_type == "token_limit_exceeded"
        assert HttpError("x", 400, error_type="sk-secret-value").error_type is None

    def test_unrecognized_type_in_body_is_dropped(self):
        key = "sk-FAKE-KEY-CANARY"
        err = HttpError("x", 400, ('{"error": {"type": "%s", "code": "%s"}}' % (key, key)).encode())
        assert err.error_type is None
        assert key not in str(err) + repr(vars(err))

    def test_detail_carries_the_error_type(self):
        err = HttpError("x", 400, b'{"error": {"type": "overloaded_error"}}')
        detail = provider_error_detail(err)
        assert detail["error_type"] == "overloaded_error"
        assert detail["code"] == "PROVIDER_UNAVAILABLE"

