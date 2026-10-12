"""Guard provider echoes before private output reaches callbacks or a graph.

Only a possible credential prefix is withheld. Safe streaming text and tool
arguments retain their original bytes, including JSON escapes and whitespace.
"""
from __future__ import annotations

from pydantic import BaseModel


class PrivateOutputError(ValueError):
    def __init__(self):
        super().__init__('Private provider response rejected')


def _escape_view(text):
    """Return decoded characters, their raw offsets and an unfinished escape."""
    chars, offsets, index = [], [], 0
    escapes = {'"':'"', '\\':'\\', '/':'/', 'b':'\b', 'f':'\f', 'n':'\n', 'r':'\r', 't':'\t'}
    while index < len(text):
        start = index
        if text[index] != '\\':
            char, index = text[index], index + 1
        elif index + 1 == len(text):
            return ''.join(chars), offsets, index
        elif text[index+1] == 'u':
            digits = text[index+2:index+6]
            if len(digits) < 4 and all(c in '0123456789abcdefABCDEF' for c in digits):
                return ''.join(chars), offsets, index
            if len(digits) != 4 or any(c not in '0123456789abcdefABCDEF' for c in digits):
                char, index = text[index], index + 1
            else:
                code = int(digits, 16)
                char, index = chr(code), index + 6
                if 0xD800 <= code <= 0xDBFF:
                    remaining = text[index:]
                    if '\\u'.startswith(remaining) or (remaining.startswith('\\u') and len(remaining) < 6):
                        return ''.join(chars), offsets, start
                    if remaining.startswith('\\u') and len(remaining) >= 6:
                        try:
                            low = int(remaining[2:6], 16)
                        except ValueError:
                            low = 0
                        if 0xDC00 <= low <= 0xDFFF:
                            char = chr(0x10000 + (code-0xD800)*0x400 + low-0xDC00)
                            index += 6
        elif text[index+1] in escapes:
            char, index = escapes[text[index+1]], index + 2
        else:
            char, index = text[index], index + 1
        chars.append(char)
        offsets.append(start)
    return ''.join(chars), offsets, None


def _escape_views(text):
    """Compose raw offsets through nested JSON escapes without rewriting output."""
    offsets = list(range(len(text)))
    current = text
    for _ in range(64):
        decoded, positions, unfinished = _escape_view(current)
        raw_unfinished = offsets[unfinished] if unfinished is not None else None
        mapped = [offsets[position] for position in positions]
        yield decoded, mapped, raw_unfinished
        if decoded == current:
            return
        current, offsets = decoded, mapped
    raise PrivateOutputError()


class PrivateOutputGuard:
    def __init__(self, credential):
        if type(credential) is not str or not credential:
            raise PrivateOutputError()
        self._credential = credential
        self._pending = {}
        self._finished = set()

    def __repr__(self):
        return '<PrivateOutputGuard>'

    def __reduce__(self):
        raise TypeError('Private output guards cannot be persisted')

    def _require_open(self):
        if self._credential is None:
            raise PrivateOutputError()

    def assert_safe(self, value, *, depth=0):
        self._require_open()
        if depth > 64:
            raise PrivateOutputError()
        if type(value) is str:
            if self._credential in value:
                raise PrivateOutputError()
            for decoded, _, _ in _escape_views(value):
                if self._credential in decoded:
                    raise PrivateOutputError()
        elif isinstance(value, BaseModel):
            self.assert_safe(value.__dict__, depth=depth+1)
        elif type(value) is dict:
            for key, item in value.items():
                self.assert_safe(key, depth=depth+1)
                self.assert_safe(item, depth=depth+1)
        elif type(value) in (list, tuple):
            for item in value:
                self.assert_safe(item, depth=depth+1)
        elif value is not None and type(value) not in (bool, int, float):
            raise PrivateOutputError()

    def _prefix_length(self, value):
        for length in range(min(len(value), len(self._credential)-1), 0, -1):
            if value.endswith(self._credential[:length]):
                return length
        return 0

    def feed(self, channel, text):
        self._require_open()
        if type(text) is not str:
            raise PrivateOutputError()
        if channel not in self._pending and len(self._pending) >= 64:
            raise PrivateOutputError()
        pending = self._pending.get(channel, '') + text
        self.assert_safe(pending)
        retained = len(pending)
        length = self._prefix_length(pending)
        if length:
            retained = len(pending)-length
        for decoded, offsets, unfinished in _escape_views(pending):
            length = self._prefix_length(decoded)
            if length:
                retained = min(retained, offsets[len(decoded)-length])
            if unfinished is not None:
                retained = min(retained, unfinished)
        self._pending[channel] = pending[retained:]
        if len(self._pending[channel]) > 12*len(self._credential)+12:
            raise PrivateOutputError()
        return pending[:retained]

    def finish(self, channel):
        self._require_open()
        pending = self._pending.pop(channel, '')
        self.assert_safe(pending)
        return pending

    def guard_chunk(self, chunk):
        self.assert_safe(chunk)
        from magic_llm.model.ModelChatStream import DeltaModel, ToolCall, FunctionCall
        # NodeLLM consumes choices by position and merges missing tool indices
        # by enumeration. Match that identity before any fragment can escape.
        for position, choice in enumerate(chunk.choices):
            if choice.delta is None:
                if choice.finish_reason is None:
                    continue
                choice.delta = DeltaModel()
            delta = choice.delta
            fields = ('content', 'refusal', 'reasoning', 'reasoning_content')
            if position in self._finished:
                if any(getattr(delta, field) for field in fields) or delta.tool_calls:
                    raise PrivateOutputError()
                continue
            for field in fields:
                value = getattr(delta, field)
                channel = (position, field)
                if value is not None:
                    setattr(delta, field, self.feed(channel, value))
                if choice.finish_reason is not None:
                    tail = self.finish(channel)
                    if tail:
                        setattr(delta, field, (getattr(delta, field) or '') + tail)
            calls = delta.tool_calls or []
            indexed_calls = {}
            for fallback_index, call in enumerate(calls):
                call_index = call.index if call.index is not None else fallback_index
                indexed_calls[call_index] = call
                if call.function is not None and call.function.arguments is not None:
                    channel = (position, 'arguments', call_index)
                    call.function.arguments = self.feed(channel, call.function.arguments)
            if choice.finish_reason is not None:
                for channel in tuple(self._pending):
                    if channel[:2] != (position, 'arguments'):
                        continue
                    tail = self.finish(channel)
                    if not tail:
                        continue
                    call = indexed_calls.get(channel[2])
                    if call is None:
                        call = ToolCall(index=channel[2], function=FunctionCall(arguments=tail))
                        calls.append(call)
                    else:
                        if call.function is None:
                            call.function = FunctionCall()
                        call.function.arguments = (call.function.arguments or '') + tail
                if calls:
                    delta.tool_calls = calls
                self._finished.add(position)
        return chunk

    def close(self):
        self._credential = None
        self._pending.clear()
        self._finished.clear()
