"""Safe provider failures shared by hosts, callbacks and HTTP transports.

Only recognized codes, recognized error types and numeric status survive.
Provider bodies can echo authentication material and prompt text, so they
are classified while they are in hand and then dropped: no body, and no
message taken from one, becomes exception text or audit data.
"""
import json

from magic_llm.exception.ChatException import ChatException, ProviderProtocolError

_AUTH = frozenset({'invalid_api_key', 'invalid_api_token', 'authentication_error',
    'invalid_authentication', 'unauthenticated', 'api_key_invalid', 'authentication'})
_QUOTA = frozenset({'insufficient_quota', 'quota_exceeded', 'credit_balance_too_low',
    'payment_required'})
_KNOWN = _AUTH | _QUOTA | {'permission_denied', 'rate_limit_exceeded', 'rate_limit_error'}

#: Error types a provider reports that say a request is too large for the
#: model. The same request fails the same way again; a smaller one may not.
CONTEXT_ERROR_TYPES = frozenset({
    'context_length_exceeded', 'token_limit_exceeded', 'payload_too_large',
    'request_too_large', 'prompt_too_long',
})

#: Typed error values kept on a failure. They name a kind of failure, never
#: its content: OpenRouter's ``error.metadata.error_type``, OpenAI's
#: ``error.type`` / ``error.code`` and Anthropic's ``error.type``.
ERROR_TYPES = CONTEXT_ERROR_TYPES | _KNOWN | frozenset({
    # OpenRouter
    'content_policy_violation', 'refusal', 'not_found', 'precondition_failed',
    'unprocessable', 'max_tokens_exceeded', 'string_too_long', 'invalid_request',
    'invalid_prompt', 'provider_unavailable', 'provider_overloaded', 'timeout',
    'server', 'unmapped', 'no_content',
    # OpenAI
    'invalid_request_error', 'server_error', 'model_not_found', 'string_above_max_length',
    # Anthropic
    'permission_error', 'not_found_error', 'api_error', 'overloaded_error',
})

# Phrases providers use for a context overflow when they give no typed code
# (a plain HTTP 400). Matched only to classify; the text is never kept.
_CONTEXT_PHRASES = (
    'maximum context length', 'context length', 'context window', 'context_length_exceeded',
    'prompt is too long', 'too many tokens', 'input is too long', 'reduce the length of the',
    'exceeds the model', 'input tokens exceed',
)

# Phrases providers use for a rejected key behind a plain HTTP 400 (Google
# answers ``400 INVALID_ARGUMENT`` with "API key not valid").
_AUTH_PHRASES = ('api key not valid', 'invalid api key', 'incorrect api key', 'api_key_invalid')

MESSAGES = {
    'PROVIDER_AUTHENTICATION_REJECTED': 'The provider rejected the API key.',
    'BYOK_CREDENTIAL_REJECTED': 'The provider rejected your API key. Enter a valid key to retry.',
    'PROVIDER_PERMISSION_DENIED': 'The provider denied access to this model or resource.',
    'PROVIDER_RATE_LIMITED': 'The provider rate limit was reached.',
    'PROVIDER_QUOTA_EXCEEDED': 'The provider account has insufficient quota or balance.',
    'PROVIDER_CONTEXT_EXCEEDED': 'The request is larger than the model accepts.',
    'PROVIDER_UNAVAILABLE': 'The provider request failed. Try again later.',
}

_MAX_BODY = 65536


def _parse(content):
    if not isinstance(content, (str, bytes)) or len(content) > _MAX_BODY:
        return None
    try:
        return json.loads(content)
    except (ValueError, UnicodeError):
        return None


def _walk(item, depth=0):
    """Every mapping in a parsed body, breadth-limited."""
    if depth > 5:
        return
    if isinstance(item, dict):
        yield item
        for child in list(item.values())[:16]:
            yield from _walk(child, depth + 1)
    elif isinstance(item, list):
        for child in item[:16]:
            yield from _walk(child, depth + 1)


def response_code(content):
    """The first recognized provider code in *content*, lower-cased, or None."""
    for item in _walk(_parse(content)):
        for name in ('code', 'type', 'status', 'reason'):
            code = item.get(name)
            if isinstance(code, str) and code.lower() in _KNOWN:
                return code.lower()
    return None


def response_type(content):
    """The provider's typed error kind in *content*, if it is a recognized one.

    OpenRouter's ``metadata.error_type`` is the most specific and wins.
    """
    found = None
    for item in _walk(_parse(content)):
        metadata = item.get('metadata')
        if isinstance(metadata, dict):
            value = metadata.get('error_type')
            if isinstance(value, str) and value.lower() in ERROR_TYPES:
                return value.lower()
        if found is None:
            for name in ('type', 'code', 'status', 'reason'):
                value = item.get(name)
                if isinstance(value, str) and value.lower() in ERROR_TYPES:
                    found = value.lower()
                    break
    return found


def _text(content):
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeError:
            return None
    if not isinstance(content, str) or len(content) > _MAX_BODY:
        return None
    return content.lower()


def mentions_context_overflow(content):
    """Whether a provider body says the request exceeded the context window."""
    text = _text(content)
    return text is not None and any(phrase in text for phrase in _CONTEXT_PHRASES)


def mentions_rejected_key(content):
    """Whether a provider body says the key was rejected, whatever the status."""
    text = _text(content)
    return text is not None and any(phrase in text for phrase in _AUTH_PHRASES)


def response_retry_after(content):
    """A numeric ``retry_after`` (seconds) in a provider body, or None."""
    for item in _walk(_parse(content)):
        for name in ('retry_after', 'retry-after', 'retryAfter'):
            value = item.get(name)
            if type(value) in (int, float) and 0 <= value <= 86400:
                return value
    return None


def failure_code(status, provider_code=None, error_type=None, context_overflow=False,
                 rejected_key=False):
    if status == 401 or provider_code in _AUTH or error_type in _AUTH or rejected_key:
        return 'PROVIDER_AUTHENTICATION_REJECTED'
    if status == 402 or provider_code in _QUOTA or error_type in _QUOTA:
        return 'PROVIDER_QUOTA_EXCEEDED'
    if status == 413 or error_type in CONTEXT_ERROR_TYPES or context_overflow:
        return 'PROVIDER_CONTEXT_EXCEEDED'
    if status == 403:
        return 'PROVIDER_PERMISSION_DENIED'
    if status == 429:
        return 'PROVIDER_RATE_LIMITED'
    return 'PROVIDER_UNAVAILABLE'

def failure_message(error):
    """Text for a failed attempt that is safe to log, store and show.

    The library's own errors (``ChatException``, ``ProviderProtocolError``)
    keep their wording; anything that may carry provider text is replaced by
    the message for its classification.
    """
    if isinstance(error, (ChatException, ProviderProtocolError)):
        return str(error)
    return provider_error_detail(error)['message']


def provider_error_detail(error, *, byok=False):
    # Runtime attempt-control exceptions wrap the original HTTP failure.
    # Follow only explicit causes, never parse exception text or raw bodies.
    current, seen = error, set()
    status, code, error_type = None, None, None
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        candidate = getattr(current, 'status_code', None)
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
        candidate = getattr(current, 'code', None)
        if isinstance(candidate, str) and candidate in MESSAGES:
            code = candidate
        candidate = getattr(current, 'error_type', None)
        if error_type is None and isinstance(candidate, str) and candidate in ERROR_TYPES:
            error_type = candidate
        # Inner Flow failures carry safe typed child codes rather than Python
        # exception causes. Use only the actual failed path, never its message.
        outcome = getattr(current, 'outcome', None)
        failure = outcome.get('error') if isinstance(outcome, dict) else None
        details = failure.get('details') if isinstance(failure, dict) else None
        if isinstance(details, dict) and isinstance(details.get('child_errors'), list):
            for child in details['child_errors'][:20]:
                if not isinstance(child, dict) or child.get('node_path') != details.get('cause_path'):
                    continue
                candidate = child.get('error_code')
                if isinstance(candidate, str) and candidate in MESSAGES:
                    code = candidate
                    break
        current = getattr(current, '__cause__', None)
    if code not in MESSAGES:
        code = failure_code(status, error_type=error_type)
    if byok and code == 'PROVIDER_AUTHENTICATION_REJECTED':
        code = 'BYOK_CREDENTIAL_REJECTED'
    return {'code': code, 'message': MESSAGES[code], 'type': 'provider_error',
            **({'provider_status': status} if status is not None else {}),
            **({'error_type': error_type} if error_type is not None else {})}
