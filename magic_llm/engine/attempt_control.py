"""Opt-in, awaited admission for actual provider retry/fallback attempts.

The host owns authorization, reservations, deadlines and durable fences. These
callbacks are correctness operations, not best-effort observer hooks. No control
object is forwarded to a provider, and shared engine configuration is untouched.
"""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from magic_llm.exception.ChatException import ChatException, RequestValidationError
from magic_llm.model.ModelChatStream import ChatMetaModel


class ProviderAttemptControlError(ChatException):
    """Terminal control/capability decision; never retried or sent to fallback."""


class ProviderStreamInterrupted(ProviderAttemptControlError):
    """An emitted response cannot be replaced by a transparent retry."""

    def __init__(self):
        super().__init__(
            'Provider stream failed after emitting response data',
            error_code='PROVIDER_STREAM_INTERRUPTED',
        )


@dataclass(frozen=True)
class ProviderAttempt:
    """Runtime identity plus private request snapshot for host admission.

    IDs are generated here, never accepted from provider/model payloads.
    request_operation_id stays stable across this call's retries and fallback;
    provider_attempt_id identifies one dispatch. Neither is a public consumer
    attemptId. Snapshot data is private and excluded from repr/log messages.
    """

    request_operation_id: str
    provider_attempt_id: str
    parent_attempt_id: str | None
    provider: str
    model: str | None
    stream: bool
    attempt_index: int
    retry_index: int
    is_fallback: bool
    messages: tuple[dict[str, Any], ...] = field(repr=False)
    generation_options: dict[str, Any] = field(repr=False)
    # Facts captured before function normalization can erase native tool types.
    # Host pricing must independently qualify features beyond local functions.
    request_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderAttemptOutcome:
    status: Literal['completed', 'failed', 'cancelled']
    semantic_output: bool
    usage: Any = field(default=None, repr=False)
    usage_uncertain: bool = False
    error_code: str | None = None


class ProviderAttemptControl(Protocol):
    async def before_attempt(self, attempt: ProviderAttempt) -> None:
        """Authorize/reserve this exact provider/model; raise to deny dispatch.

        The host must check cancellation, fence, deadline and cumulative limits
        on every call. No provider work starts unless this await succeeds.
        """

    async def after_attempt(
        self, attempt: ProviderAttempt, outcome: ProviderAttemptOutcome,
    ) -> None:
        """Reconcile/release admission exactly once for a started attempt.

        Unknown usage is not a refund. Persist cleanup under cancellation as
        required by the host. Failure stops this logical request without retry.
        """


def reject_sync_control(control: ProviderAttemptControl | None) -> None:
    if control is not None:
        raise ProviderAttemptControlError(
            'Provider attempt control requires an async entry point',
            error_code='PROVIDER_ATTEMPT_CONTROL_UNSUPPORTED',
        )


def require_attempt_capability(method: Any) -> None:
    if getattr(method, '_provider_attempt_control_version', None) != 1:
        raise ProviderAttemptControlError(
            'Provider method does not support mandatory attempt control',
            error_code='PROVIDER_ATTEMPT_CONTROL_UNSUPPORTED',
        )


def has_semantic_output(chunk: Any) -> bool:
    """Conservative boundary: include replay state and non-text semantics."""
    if any(getattr(chunk, key, None) for key in ('responses_output', 'gemini_parts', 'extras')):
        return True
    for choice in getattr(chunk, 'choices', ()) or ():
        if getattr(choice, 'finish_reason', None) or getattr(choice, 'logprobs', None):
            return True
        delta = getattr(choice, 'delta', None)
        if delta is None:
            continue
        if any(getattr(delta, key, None) for key in (
            'content', 'tool_calls', 'refusal', 'annotations',
            'reasoning_content', 'reasoning', 'reasoning_details',
        )):
            return True
    return False


def _request_options(engine: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    # Never copy transport credentials or callback/control objects into a
    # request-admission snapshot. It is private context, not public telemetry.
    excluded = {'headers', 'authorization', 'api_key', 'private_key', 'secret',
                'password', 'token', 'callback', 'fallback', 'executor',
                'provider_attempt_control'}

    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()
                    if str(key).lower() not in excluded
                    and not any(part in str(key).lower() for part in
                                ('secret', 'access_key', 'auth_token', 'api_key'))}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        return copy.deepcopy(value)

    options = {**getattr(engine, 'kwargs', {}), **kwargs}
    if options.get('tools'):
        from magic_llm.engine.tooling import normalize_openai_tools
        options['tools'] = normalize_openai_tools(options['tools'])
    return sanitize(options)


def _providers(engine: Any, func: Any, stream: bool):
    """Validate the whole fallback chain before the first external operation."""
    result = []
    seen = set()
    method_name = 'async_stream_generate' if stream else 'async_generate'
    while True:
        if id(engine) in seen or len(result) >= 16:
            raise ProviderAttemptControlError(
                'Provider fallback chain is cyclic or too deep',
                error_code='PROVIDER_ATTEMPT_CONTROL_UNSUPPORTED',
            )
        seen.add(id(engine))
        attempts = getattr(getattr(engine, 'retry_config', None), 'attempts', None)
        delay = getattr(getattr(engine, 'retry_config', None), 'delay', None)
        import math
        if (type(attempts) is not int or attempts < 1 or
                not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0):
            raise ProviderAttemptControlError(
                'Controlled provider retries require finite positive attempts and finite delay',
                error_code='PROVIDER_ATTEMPT_CONTROL_UNSUPPORTED',
            )
        result.append((engine, func, attempts, delay))
        fallback = getattr(engine, 'fallback', None)
        if fallback is None:
            return result
        engine = getattr(fallback, 'llm', None)
        method = getattr(engine, method_name, None)
        require_attempt_capability(method)
        func = getattr(method, '_provider_attempt_raw')


def _attempt_model(engine, options, stream):
    """Describe the model the adapter will actually dispatch, without I/O."""
    configured_model = getattr(engine, 'model', None)
    requested_model = options.get('model', configured_model)
    if str(getattr(engine, 'engine', '')).lower() not in {'google', 'gemini'}:
        return requested_model
    # Gemini binds the model into its URL at construction; call-time model
    # options do not change that endpoint. Never admit a different identity.
    endpoint = getattr(engine, 'url_stream' if stream else 'url', None)
    endpoint_model = configured_model
    if endpoint is not None:
        suffix = ':streamGenerateContent' if stream else ':generateContent'
        try:
            path = unquote(urlsplit(endpoint).path)
            if '/models/' not in path or not path.endswith(suffix):
                raise ValueError('unrecognized model endpoint')
            endpoint_model = path.split('/models/', 1)[1][:-len(suffix)]
            if not endpoint_model:
                raise ValueError('missing endpoint model')
        except (TypeError, ValueError) as error:
            raise ProviderAttemptControlError(
                'Google model endpoint cannot be verified for attempt admission',
                error_code='PROVIDER_ATTEMPT_MODEL_MISMATCH',
            ) from error
    if requested_model != endpoint_model or configured_model != endpoint_model:
        raise ProviderAttemptControlError(
            'Google model configuration/override disagrees with its bound endpoint',
            error_code='PROVIDER_ATTEMPT_MODEL_MISMATCH',
        )
    return endpoint_model


def _new_attempt(engine, chat, kwargs, operation_id, parent_id,
                 stream, index, retry_index, is_fallback):
    raw_options = {**getattr(engine, 'kwargs', {}), **kwargs}
    native_tools = any(isinstance(tool, dict) and tool.get('type') not in (None, 'function')
                       for tool in (raw_options.get('tools') or ()))
    options = _request_options(engine, kwargs)
    return ProviderAttempt(
        request_operation_id=operation_id,
        provider_attempt_id=str(uuid4()),
        parent_attempt_id=parent_id,
        provider=str(getattr(engine, 'engine', type(engine).__name__)),
        model=_attempt_model(engine, options, stream),
        stream=stream, attempt_index=index, retry_index=retry_index,
        is_fallback=is_fallback,
        messages=tuple(copy.deepcopy(chat.messages)),
        generation_options=options,
        request_features=('provider_native_tools',) if native_tools else (),
    )


def _outcome(error, semantic, usage):
    # Model defaults contain a zero-filled UsageModel even when the provider
    # supplied no usage. A successful response is not evidence of zero cost.
    measured_usage = usage is not None and any(
        getattr(usage, key, 0) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    return ProviderAttemptOutcome(
        status=('cancelled' if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else 'failed' if error is not None else 'completed'),
        semantic_output=semantic,
        usage=copy.deepcopy(usage),
        usage_uncertain=not isinstance(error, RequestValidationError)
        and (error is not None or not measured_usage),
        error_code=(getattr(error, 'error_code', None) or type(error).__name__)
        if error is not None else None,
    )


async def _observe(engine, chat, attempt, outcome, content, started):
    # Existing callback implementation provides its own observer isolation and
    # Skills projection. It cannot authorize retries or replace control results.
    elapsed = time.monotonic() - started
    await engine._execute_callback(
        chat, content, outcome.usage, attempt.model,
        ChatMetaModel(TTF=elapsed, TPS=0, status=outcome.status),
    )


def _terminal_error(error, semantic):
    if not isinstance(error, Exception):
        raise error
    if isinstance(error, (RequestValidationError, ProviderAttemptControlError)):
        raise error
    if semantic:
        raise ProviderStreamInterrupted() from error


async def controlled_stream(engine, func, chat, kwargs, control):
    chain = _providers(engine, func, True)
    operation_id, parent_id, index = str(uuid4()), None, 0
    last_error = None
    for provider_index, (current, raw, attempts, delay) in enumerate(chain):
        for retry_index in range(1, attempts + 1):
            index += 1
            attempt = _new_attempt(current, chat, kwargs, operation_id, parent_id,
                                   True, index, retry_index, provider_index > 0)
            # Outside the provider-error handler: any admission denial is final.
            await control.before_attempt(attempt)
            parent_id = attempt.provider_attempt_id
            error, usage, semantic, content = None, None, False, ''
            started = time.monotonic()
            source = None
            try:
                source = raw(current, chat, **kwargs)
                async for chunk in source:
                    if getattr(chunk, 'usage', None) is not None:
                        candidate = chunk.usage
                        if usage is None or any(getattr(candidate, key, 0) for key in
                                                ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                            usage = copy.deepcopy(candidate)
                    semantic = semantic or has_semantic_output(chunk)
                    for choice in getattr(chunk, 'choices', ()) or ():
                        content += getattr(getattr(choice, 'delta', None), 'content', None) or ''
                    yield chunk
            except BaseException as caught:
                error = caught
            finally:
                if source is not None:
                    try:
                        await source.aclose()
                    except BaseException as caught:
                        if error is None:
                            error = caught
                outcome = _outcome(error, semantic, usage)
                await control.after_attempt(attempt, outcome)
            await _observe(current, chat, attempt, outcome, content, started)
            if error is None:
                return
            _terminal_error(error, semantic)
            last_error = error
            if retry_index < attempts:
                await asyncio.sleep(delay)
    raise ProviderAttemptControlError(
        'Provider generation exhausted its admitted attempts',
        error_code='PROVIDER_ATTEMPTS_EXHAUSTED',
    ) from last_error


async def controlled_generate(engine, func, chat, kwargs, control):
    chain = _providers(engine, func, False)
    operation_id, parent_id, index = str(uuid4()), None, 0
    last_error = None
    for provider_index, (current, raw, attempts, delay) in enumerate(chain):
        for retry_index in range(1, attempts + 1):
            index += 1
            attempt = _new_attempt(current, chat, kwargs, operation_id, parent_id,
                                   False, index, retry_index, provider_index > 0)
            await control.before_attempt(attempt)
            parent_id = attempt.provider_attempt_id
            started = time.monotonic()
            error, response = None, None
            try:
                response = await raw(current, chat, **kwargs)
            except BaseException as caught:
                error = caught
            outcome = _outcome(error, response is not None, getattr(response, 'usage', None))
            await control.after_attempt(attempt, outcome)
            await _observe(current, chat, attempt, outcome,
                           getattr(response, 'content', None), started)
            if error is None:
                return response
            _terminal_error(error, False)
            last_error = error
            if retry_index < attempts:
                await asyncio.sleep(delay)
    raise ProviderAttemptControlError(
        'Provider generation exhausted its admitted attempts',
        error_code='PROVIDER_ATTEMPTS_EXHAUSTED',
    ) from last_error
