"""Safe provider failures shared by hosts, callbacks and HTTP transports.

Only recognized codes and numeric status survive. Provider bodies can echo
authentication material and must never become exception text or audit data.
"""
import json

_AUTH = frozenset({'invalid_api_key', 'invalid_api_token', 'authentication_error',
    'invalid_authentication', 'unauthenticated', 'api_key_invalid'})
_QUOTA = frozenset({'insufficient_quota', 'quota_exceeded', 'credit_balance_too_low'})
_KNOWN = _AUTH | _QUOTA | {'permission_denied', 'rate_limit_exceeded', 'rate_limit_error'}
MESSAGES = {
    'PROVIDER_AUTHENTICATION_REJECTED': 'The provider rejected the API key.',
    'BYOK_CREDENTIAL_REJECTED': 'The provider rejected your API key. Enter a valid key to retry.',
    'PROVIDER_PERMISSION_DENIED': 'The provider denied access to this model or resource.',
    'PROVIDER_RATE_LIMITED': 'The provider rate limit was reached.',
    'PROVIDER_QUOTA_EXCEEDED': 'The provider account has insufficient quota or balance.',
    'PROVIDER_UNAVAILABLE': 'The provider request failed. Try again later.',
}

def response_code(content):
    if not isinstance(content, (str, bytes)) or len(content) > 65536:
        return None
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError):
        return None
    def find(item, depth=0):
        if depth > 5:
            return None
        if isinstance(item, dict):
            for name in ('code', 'type', 'status', 'reason'):
                code = item.get(name)
                if isinstance(code, str) and code.lower() in _KNOWN:
                    return code.lower()
            for child in list(item.values())[:16]:
                result = find(child, depth + 1)
                if result:
                    return result
        elif isinstance(item, list):
            for child in item[:16]:
                result = find(child, depth + 1)
                if result:
                    return result
        return None
    return find(value)

def failure_code(status, provider_code=None):
    if status == 401 or provider_code in _AUTH:
        return 'PROVIDER_AUTHENTICATION_REJECTED'
    if provider_code in _QUOTA:
        return 'PROVIDER_QUOTA_EXCEEDED'
    if status == 403:
        return 'PROVIDER_PERMISSION_DENIED'
    if status == 429:
        return 'PROVIDER_RATE_LIMITED'
    return 'PROVIDER_UNAVAILABLE'

def provider_error_detail(error, *, byok=False):
    # Runtime attempt-control exceptions wrap the original HTTP failure.
    # Follow only explicit causes, never parse exception text or raw bodies.
    current, seen = error, set()
    status, code = None, None
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
        code = failure_code(status)
    if byok and code == 'PROVIDER_AUTHENTICATION_REJECTED':
        code = 'BYOK_CREDENTIAL_REJECTED'
    return {'code': code, 'message': MESSAGES[code], 'type': 'provider_error',
            **({'provider_status': status} if status is not None else {})}
