"""Tests for HttpError exception class (magic_llm.util.http)."""

import pytest

from magic_llm.util.http import HttpError


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
        body = b'{"error": "not found"}'
        err = HttpError("not found", status_code=404, response_content=body)
        assert str(err) == "not found"
        assert err.status_code == 404
        assert err.response_content == body

    def test_response_content_only(self):
        body = b'bad response'
        err = HttpError("bad", response_content=body)
        assert err.status_code is None
        assert err.response_content == body

    def test_is_exception(self):
        err = HttpError("test")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(HttpError, match="boom"):
            raise HttpError("boom", status_code=500)

    def test_args_attribute_accessible(self):
        err = HttpError("err", status_code=503)
        assert err.args == ("err",)
