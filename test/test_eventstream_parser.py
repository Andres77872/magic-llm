"""Tests for AWS EventStream binary parser.

Constructs synthetic EventStream frames following the AWS specification:
    [4B: total_length][4B: headers_length][4B: prelude_crc]
    [headers][payload][4B: message_crc]
"""
import binascii
import json
import struct

import pytest

from magic_llm.util.eventstream import AWSEventStreamParser, EventStreamParseError


def _build_eventstream_frame(headers: dict, payload: bytes) -> bytes:
    """Build a valid AWS EventStream binary frame.

    Args:
        headers: Dict of header name -> (value_type, value) tuples
        payload: Raw payload bytes

    Returns:
        Complete EventStream frame bytes
    """
    # Encode headers
    header_bytes = b""
    for name, (value_type, value) in headers.items():
        name_encoded = name.encode("utf-8")
        header_bytes += struct.pack("B", len(name_encoded))
        header_bytes += name_encoded
        header_bytes += struct.pack("B", value_type)
        if value_type == 0:  # bool false
            pass
        elif value_type == 1:  # bool true
            pass
        elif value_type == 7:  # byte array
            header_bytes += struct.pack(">H", len(value))
            header_bytes += value
        elif value_type == 8:  # string
            if isinstance(value, str):
                value = value.encode("utf-8")
            header_bytes += struct.pack(">H", len(value))
            header_bytes += value

    headers_length = len(header_bytes)
    # total_length = prelude(8) + headers + prelude_crc(4) + payload + message_crc(4)
    total_length = 8 + headers_length + 4 + len(payload) + 4

    # Build prelude (total_length + headers_length)
    prelude = struct.pack(">II", total_length, headers_length)

    # Prelude CRC: CRC32 of the 8-byte prelude
    prelude_crc = binascii.crc32(prelude) & 0xFFFFFFFF

    # Message CRC: CRC32 of everything except the last 4 bytes
    message_body = prelude + struct.pack(">I", prelude_crc) + header_bytes + payload
    message_crc = binascii.crc32(message_body) & 0xFFFFFFFF

    return message_body + struct.pack(">I", message_crc)


def _string_header(name: str, value: str) -> tuple:
    """Helper to create a string header."""
    return (7, value.encode("utf-8"))


def _bool_header_true(name: str) -> tuple:
    """Helper to create a boolean true header."""
    return (1, b"")


def _bool_header_false(name: str) -> tuple:
    """Helper to create a boolean false header."""
    return (0, b"")


class TestEventStreamParserValidFrames:
    """Test parsing of valid EventStream frames."""

    def test_parses_single_chunk_event(self):
        """Parser decodes a valid chunk event with JSON payload."""
        payload_dict = {
            "contentBlockDelta": {"delta": {"text": "Hello"}},
            "index": 0,
        }
        payload_bytes = json.dumps(payload_dict).encode("utf-8")

        frame = _build_eventstream_frame(
            headers={
                ":message-type": _string_header(":message-type", "event"),
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload_bytes,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "chunk"
        assert events[0][":content-type"] == "application/json"
        assert events[0]["payload"] == payload_dict

    def test_parses_multiple_events_in_one_feed(self):
        """Parser handles multiple events fed at once."""
        payload1 = json.dumps({"contentBlockDelta": {"delta": {"text": "A"}}}).encode()
        payload2 = json.dumps({"contentBlockDelta": {"delta": {"text": "B"}}}).encode()

        frame1 = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload1,
        )
        frame2 = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload2,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame1 + frame2))

        assert len(events) == 2
        assert events[0]["payload"]["contentBlockDelta"]["delta"]["text"] == "A"
        assert events[1]["payload"]["contentBlockDelta"]["delta"]["text"] == "B"

    def test_parses_metadata_event(self):
        """Parser handles metadata events."""
        payload = json.dumps({
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 10,
                "outputTokenCount": 20,
            }
        }).encode()

        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "metadata"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "metadata"
        assert events[0]["payload"]["amazon-bedrock-invocationMetrics"]["inputTokenCount"] == 10

    def test_parses_exception_event(self):
        """Parser handles exception events."""
        payload = json.dumps({
            "message": "ThrottlingException",
            "type": "ThrottlingException",
        }).encode()

        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "exception"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "exception"
        assert events[0]["payload"]["message"] == "ThrottlingException"

    def test_parses_event_with_empty_headers(self):
        """Parser handles events with no headers."""
        payload = b"raw payload"
        frame = _build_eventstream_frame(headers={}, payload=payload)

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0]["payload"] == payload


class TestEventStreamParserPartialFeed:
    """Test partial message buffering."""

    def test_partial_feed_yields_nothing(self):
        """Feeding half a message yields no events."""
        payload = json.dumps({"test": True}).encode()
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        half = len(frame) // 2
        events = list(parser.feed(frame[:half]))
        assert len(events) == 0

    def test_rest_of_feed_yields_events(self):
        """Feeding the rest of a partial message yields events."""
        payload = json.dumps({"test": True}).encode()
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        half = len(frame) // 2
        events1 = list(parser.feed(frame[:half]))
        assert len(events1) == 0

        events2 = list(parser.feed(frame[half:]))
        assert len(events2) == 1
        assert events2[0]["payload"] == {"test": True}

    def test_byte_by_byte_feed(self):
        """Feeding one byte at a time still works."""
        payload = json.dumps({"x": 1}).encode()
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        all_events = []
        for i in range(len(frame)):
            all_events.extend(parser.feed(frame[i:i+1]))

        assert len(all_events) == 1
        assert all_events[0]["payload"] == {"x": 1}


class TestEventStreamParserMalformedFrames:
    """Test handling of malformed frames."""

    def test_invalid_prelude_crc_raises(self):
        """Frame with corrupted prelude CRC raises EventStreamParseError."""
        payload = json.dumps({"test": True}).encode()
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        # Corrupt the prelude CRC (it's at offset 8, right after the 8-byte prelude)
        prelude_crc_offset = 8
        corrupted = bytearray(frame)
        corrupted[prelude_crc_offset] ^= 0xFF  # Flip bits

        parser = AWSEventStreamParser()
        with pytest.raises(EventStreamParseError, match="Prelude CRC mismatch"):
            list(parser.feed(bytes(corrupted)))

    def test_invalid_message_crc_raises(self):
        """Frame with corrupted message CRC raises EventStreamParseError."""
        payload = json.dumps({"test": True}).encode()
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload,
        )

        # Corrupt the last 4 bytes (message CRC)
        corrupted = bytearray(frame)
        corrupted[-1] ^= 0xFF

        parser = AWSEventStreamParser()
        with pytest.raises(EventStreamParseError, match="Message CRC mismatch"):
            list(parser.feed(bytes(corrupted)))

    def test_truncated_frame_yields_nothing(self):
        """Truncated frame (too short for minimum) yields nothing."""
        parser = AWSEventStreamParser()
        events = list(parser.feed(b"\x00\x00"))  # Only 2 bytes, need at least 8
        assert len(events) == 0

    def test_invalid_total_length_raises(self):
        """Frame with total_length < 16 raises error."""
        # total_length=8, headers_length=0 — too small
        frame = struct.pack(">II", 8, 0) + b"\x00" * 8
        # Add a fake CRC
        prelude_crc = binascii.crc32(frame[:8]) & 0xFFFFFFFF
        frame += struct.pack(">I", prelude_crc)
        message_crc = binascii.crc32(frame) & 0xFFFFFFFF
        frame += struct.pack(">I", message_crc)

        parser = AWSEventStreamParser()
        with pytest.raises(EventStreamParseError, match="total_length.*too small"):
            list(parser.feed(frame))

    def test_non_json_payload_returns_raw_bytes(self):
        """Non-JSON payload is returned as raw bytes."""
        payload = b"not json at all"
        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/octet-stream"),
            },
            payload=payload,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0]["payload"] == payload


class TestEventStreamParserBedrockRealistic:
    """Test with realistic Bedrock streaming scenarios."""

    def test_nova_streaming_chunk(self):
        """Parser handles Nova-style contentBlockDelta event."""
        # Simulate the Bedrock chunk.bytes payload (base64-encoded JSON)
        import base64
        provider_event = {
            "contentBlockDelta": {
                "delta": {"text": "Hello, world!"}
            },
            "index": 0,
        }
        # In real Bedrock, the EventStream payload contains:
        # {"bytes": "<base64-encoded-provider-event>", "amazon-bedrock-invocationMetrics": {...}}
        bedrock_payload = {
            "bytes": base64.b64encode(json.dumps(provider_event).encode()).decode(),
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 5,
                "outputTokenCount": 3,
            },
        }
        payload_bytes = json.dumps(bedrock_payload).encode()

        frame = _build_eventstream_frame(
            headers={
                ":message-type": _string_header(":message-type", "event"),
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload_bytes,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "chunk"
        # The 'bytes' field should be present (base64 string)
        assert "bytes" in events[0]["payload"]
        # And invocation metrics
        assert "amazon-bedrock-invocationMetrics" in events[0]["payload"]

    def test_titan_streaming_chunk(self):
        """Parser handles Titan-style outputText event."""
        import base64
        provider_event = {
            "outputText": "Hello from Titan",
            "index": 0,
            "completionReason": None,
        }
        bedrock_payload = {
            "bytes": base64.b64encode(json.dumps(provider_event).encode()).decode(),
        }
        payload_bytes = json.dumps(bedrock_payload).encode()

        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload_bytes,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "chunk"
        assert "bytes" in events[0]["payload"]

    def test_end_of_stream_event(self):
        """Parser handles end-of-stream (messageStop) events."""
        import base64
        provider_event = {
            "messageStop": {"stopReason": "end_turn"},
        }
        bedrock_payload = {
            "bytes": base64.b64encode(json.dumps(provider_event).encode()).decode(),
        }
        payload_bytes = json.dumps(bedrock_payload).encode()

        frame = _build_eventstream_frame(
            headers={
                ":event-type": _string_header(":event-type", "chunk"),
                ":content-type": _string_header(":content-type", "application/json"),
            },
            payload=payload_bytes,
        )

        parser = AWSEventStreamParser()
        events = list(parser.feed(frame))

        assert len(events) == 1
        assert events[0][":event-type"] == "chunk"
