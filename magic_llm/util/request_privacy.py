"""Metadata-only provider diagnostics for explicitly protected conversations."""
import json


def protected_payload_summary(provider, payload):
    messages = payload.get('messages', payload.get('contents', payload.get('input', [])))
    return json.dumps({
        'marker': 'MAGIC_LLM_DEBUG_PAYLOAD_REDACTED',
        'provider': provider.__class__.__name__,
        'model': payload.get('model', getattr(provider, 'model', None)),
        'stream': bool(payload.get('stream')),
        'message_count': len(messages) if isinstance(messages, list) else 0,
        'tool_count': len(payload.get('tools') or []),
        'payload_chars': len(json.dumps(payload, default=str)),
    })
