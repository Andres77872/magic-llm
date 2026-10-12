"""Immutable execution limits applied to the final provider request.

This module has no account, catalog or billing policy. The caller supplies an
already authorized model and limits. Each call receives its own chat copy.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Callable

from magic_llm.exception.ChatException import ChatException, RequestValidationError
from magic_llm.util.tokenizer import from_openai


def _reject(code):
    raise RequestValidationError(ChatException('Execution request is outside its limits', error_code=code))


@dataclass(frozen=True)
class PrivateRequestLimits:
    model: str
    max_input_tokens: int
    max_output_tokens: int
    max_messages: int
    timeout_seconds: float
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    truncation_strategy: str = 'token_budget'
    authorize_request: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.authorize_request is not None and not callable(self.authorize_request):
            raise ValueError('Execution request authorization must be callable')
        if not isinstance(self.model, str) or not self.model:
            raise ValueError('Execution model is required')
        if self.truncation_strategy not in {'tail', 'token_budget'}:
            raise ValueError('Execution truncation strategy is invalid')
        for name in ('max_input_tokens', 'max_output_tokens', 'max_messages'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('Execution limits must be positive integers')
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 600):
            raise ValueError('Execution timeout is invalid')
        for name, low, high in (('temperature', 0, 2), ('top_p', 0, 1)):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                      or not low <= value <= high or name == 'top_p' and value == 0):
                raise ValueError('Execution sampling parameter is invalid')

    def prepare(self, chat, options):
        # This authority remains in memory. It can invalidate clients already
        # allocated for later roles when any earlier stage fails or closes.
        if self.authorize_request is not None and self.authorize_request() is not True:
            _reject('EXECUTION_REQUEST_UNAVAILABLE')
        overrides = {'model', 'messages', 'input', 'instructions', 'headers', 'api_key', 'private_key',
                     'authorization', 'base_url', 'endpoint', 'extra_body', 'extra_headers',
                     'proxy', 'proxy_auth', 'ssl', 'server_hostname', 'auth',
                     'previous_response_id', 'conversation', 'background', 'cache_control'}
        if overrides.intersection(options):
            _reject('EXECUTION_OVERRIDE_FORBIDDEN')
        options = dict(options)
        if options.get('n', 1) != 1:
            _reject('EXECUTION_MULTIPLE_OUTPUTS_FORBIDDEN')
        cap = self.max_output_tokens
        for name in ('max_tokens', 'max_output_tokens', 'max_completion_tokens'):
            value = options.pop(name, None)
            if value is not None:
                if type(value) is not int or value < 1:
                    _reject('EXECUTION_OUTPUT_LIMIT_INVALID')
                cap = min(cap, value)
        options['max_tokens'] = cap
        # Profile settings replace graph/runtime sampling, including aliases.
        options.pop('reasoning', None)
        for name in ('temperature', 'top_p', 'reasoning_effort'):
            options.pop(name, None)
            if (value := getattr(self, name)) is not None:
                options[name] = value
        requested_timeout = options.pop('timeout', self.timeout_seconds)
        if (type(requested_timeout) not in (int, float) or not math.isfinite(requested_timeout)
                or requested_timeout <= 0):
            _reject('EXECUTION_TIMEOUT_INVALID')
        timeout = min(requested_timeout, self.timeout_seconds)
        options['store'] = False
        # Keep existing mandatory validators and observer projection; never
        # mutate a chat shared with the graph, another role or a loop replay.
        prepared = copy.copy(chat)
        prepared.messages = copy.deepcopy(chat.messages)
        prepared.max_input_tokens = min(chat.max_input_tokens or self.max_input_tokens, self.max_input_tokens)
        # The workflow prepares its context window with the frozen strategy.
        # A provider boundary must preserve that resulting context (especially
        # current tool exchanges) and fail on overflow, never silently trim it.
        prepared.require_complete_context()
        original_guard = getattr(chat, '_provider_payload_guard', None)
        def guard(payload):
            self.validate(payload, cap)
            if original_guard is not None:
                return original_guard(payload)
        prepared.set_provider_payload_guard(guard)
        return prepared, options, timeout

    def validate(self, payload, cap):
        if payload.get('model') != self.model:
            _reject('EXECUTION_MODEL_CHANGED')
        limits = [payload[name] for name in ('max_tokens', 'max_output_tokens', 'max_completion_tokens')
                  if name in payload]
        if not limits or any(type(value) is not int or not 0 < value <= cap for value in limits):
            _reject('EXECUTION_OUTPUT_LIMIT_INVALID')
        if any(name in payload for name in ('private_transport', 'private_limits', 'api_key', 'private_key',
                                            'headers', 'authorization', 'base_url', 'extra_body', 'timeout')):
            _reject('EXECUTION_TRANSPORT_IN_PAYLOAD')
        for name in ('temperature', 'top_p'):
            if payload.get(name) != getattr(self, name):
                _reject('EXECUTION_SAMPLING_CHANGED')
        effort = payload.get('reasoning_effort', (payload.get('reasoning') or {}).get('effort'))
        if effort != self.reasoning_effort:
            _reject('EXECUTION_SAMPLING_CHANGED')
        messages = payload.get('messages', payload.get('input', []))
        if not isinstance(messages, list) or len(messages) > self.max_messages:
            _reject('EXECUTION_MESSAGE_LIMIT_EXCEEDED')
        # Count the mapped context and tool schemas. This is an application
        # estimator, not a promise of the provider's exact tokenizer count.
        context = {name: payload[name] for name in ('messages', 'input', 'instructions', 'tools', 'response_format', 'text')
                   if name in payload}
        try:
            estimate = len(from_openai(json.dumps(context, ensure_ascii=False, allow_nan=False)))
        except (ValueError, TypeError):
            _reject('EXECUTION_CONTEXT_INVALID')
        if estimate > self.max_input_tokens:
            _reject('EXECUTION_INPUT_LIMIT_EXCEEDED')
