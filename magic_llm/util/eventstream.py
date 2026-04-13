"""
AWS EventStream binary parser for Bedrock streaming responses.

Parses the AWS EventStream binary framing format:
    [4B: total_length][4B: headers_length][4B: prelude_crc]
    [headers][payload][4B: message_crc]

The prelude CRC (at offset 8) covers bytes [0:8] (total_length + headers_length).
The message CRC (last 4 bytes) covers bytes [0:total_length-4].

Usage:
    parser = AWSEventStreamParser()
    for event in parser.feed(raw_bytes_chunk):
        event_type = event[':event-type']  # 'chunk', 'exception', 'metadata'
        payload = event['payload']  # parsed JSON dict or raw bytes
"""

import binascii
import json
import logging
import struct
from typing import Iterator

logger = logging.getLogger(__name__)

# EventStream header value types — per AWS Smithy EventStream spec
# https://awslabs.github.io/smithy/1.0/spec/core/stream-traits.html#eventheader
_HEADER_TYPE_BOOL_TRUE = 0
_HEADER_TYPE_BOOL_FALSE = 1
_HEADER_TYPE_BYTE = 2
_HEADER_TYPE_SHORT = 3
_HEADER_TYPE_INTEGER = 4
_HEADER_TYPE_LONG = 5
_HEADER_TYPE_BYTE_ARRAY = 6
_HEADER_TYPE_STRING = 7


class EventStreamParseError(Exception):
    """Raised when EventStream binary parsing fails."""
    pass


class AWSEventStreamParser:
    """
    Parses AWS EventStream binary framing into event dicts.

    Maintains an internal buffer to handle partial messages that span
    multiple HTTP chunks.
    """

    def __init__(self):
        self._buffer = b''

    def feed(self, data: bytes) -> Iterator[dict]:
        """
        Feed raw bytes and yield complete parsed events.

        Handles partial messages by buffering incomplete frames.
        Validates CRC32 checksums for both prelude and message integrity.

        Args:
            data: Raw bytes from the HTTP stream

        Yields:
            dict with ':event-type', ':content-type', and 'payload' keys
        """
        self._buffer += data

        while len(self._buffer) >= 8:
            # Read prelude: total_length (4B) + headers_length (4B)
            total_length, headers_length = struct.unpack('>II', self._buffer[:8])

            # Sanity check: total_length must be at least 16 bytes
            # (8 prelude + 4 prelude_crc + 4 message_crc minimum)
            if total_length < 16:
                raise EventStreamParseError(
                    f"Invalid EventStream frame: total_length={total_length} is too small"
                )

            if len(self._buffer) < total_length:
                # Incomplete message, wait for more data
                break

            message = self._buffer[:total_length]
            self._buffer = self._buffer[total_length:]

            # Validate CRC32 checksums
            self._validate_crc32(message, headers_length)

            # Parse headers (after prelude[8] + prelude_crc[4] = offset 12)
            header_start = 12
            header_end = 12 + headers_length
            headers = self._parse_headers(message[header_start:header_end])

            # Parse payload (after headers, before message_crc)
            payload_start = header_end
            payload_end = total_length - 4  # skip message_crc
            payload_bytes = message[payload_start:payload_end]

            event_type = headers.get(':event-type', '')
            content_type = headers.get(':content-type', '')

            event = {
                ':event-type': event_type,
                ':content-type': content_type,
            }

            # Parse JSON payload if content-type is application/json
            if content_type == 'application/json' and payload_bytes:
                try:
                    event['payload'] = json.loads(payload_bytes.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse JSON payload for event type '{event_type}': {e}")
                    event['payload'] = payload_bytes
            elif payload_bytes:
                event['payload'] = payload_bytes

            yield event

    def _validate_crc32(self, message: bytes, headers_length: int) -> None:
        """
        Validate CRC32 checksums for prelude and message integrity.

        Args:
            message: Complete message bytes
            headers_length: Length of the headers section

        Raises:
            EventStreamParseError: If CRC validation fails
        """
        total_length = len(message)

        # Validate prelude CRC (CRC32 of first 8 bytes: total_length + headers_length)
        # Prelude CRC is always at offset 8 (immediately after the 8-byte prelude)
        prelude_crc_offset = 8
        expected_prelude_crc = struct.unpack('>I', message[prelude_crc_offset:prelude_crc_offset + 4])[0]
        actual_prelude_crc = binascii.crc32(message[:8]) & 0xFFFFFFFF

        if expected_prelude_crc != actual_prelude_crc:
            raise EventStreamParseError(
                f"Prelude CRC mismatch: expected={expected_prelude_crc:#010x}, "
                f"actual={actual_prelude_crc:#010x}"
            )

        # Validate message CRC (CRC32 of entire message except last 4 bytes)
        expected_message_crc = struct.unpack('>I', message[total_length - 4:total_length])[0]
        actual_message_crc = binascii.crc32(message[:total_length - 4]) & 0xFFFFFFFF

        if expected_message_crc != actual_message_crc:
            raise EventStreamParseError(
                f"Message CRC mismatch: expected={expected_message_crc:#010x}, "
                f"actual={actual_message_crc:#010x}"
            )

    def _parse_headers(self, header_bytes: bytes) -> dict[str, str | bytes]:
        """
        Parse EventStream binary-encoded headers.

        Format per header:
            [1B name_length][name bytes][1B value_type][value]

        Value encoding depends on value_type:
            0: boolean true (no value bytes)
            1: boolean false (no value bytes)
            2: byte (1B)
            3: short (2B, big-endian)
            4: integer (4B, big-endian)
            5: long (8B, big-endian)
            6: byte array (2B length + bytes)
            7: UTF-8 string (2B length + bytes)

        Args:
            header_bytes: Raw header bytes (between prelude and prelude_crc)

        Returns:
            Dict mapping header names to their decoded values
        """
        headers: dict[str, str | bytes] = {}
        offset = 0

        while offset < len(header_bytes):
            # Read header name
            name_len = header_bytes[offset]
            offset += 1
            name = header_bytes[offset:offset + name_len].decode('utf-8')
            offset += name_len

            # Read value type
            value_type = header_bytes[offset]
            offset += 1

            if value_type == _HEADER_TYPE_BOOL_TRUE:
                headers[name] = b'true'
            elif value_type == _HEADER_TYPE_BOOL_FALSE:
                headers[name] = b'false'
            elif value_type == _HEADER_TYPE_BYTE:
                headers[name] = struct.pack('b', header_bytes[offset])
                offset += 1
            elif value_type == _HEADER_TYPE_SHORT:
                headers[name] = struct.pack('>h', struct.unpack('>h', header_bytes[offset:offset + 2])[0])
                offset += 2
            elif value_type == _HEADER_TYPE_INTEGER:
                headers[name] = struct.pack('>i', struct.unpack('>i', header_bytes[offset:offset + 4])[0])
                offset += 4
            elif value_type == _HEADER_TYPE_LONG:
                headers[name] = struct.pack('>q', struct.unpack('>q', header_bytes[offset:offset + 8])[0])
                offset += 8
            elif value_type == _HEADER_TYPE_BYTE_ARRAY:
                val_len = struct.unpack('>H', header_bytes[offset:offset + 2])[0]
                offset += 2
                headers[name] = header_bytes[offset:offset + val_len]
                offset += val_len
            elif value_type == _HEADER_TYPE_STRING:
                val_len = struct.unpack('>H', header_bytes[offset:offset + 2])[0]
                offset += 2
                headers[name] = header_bytes[offset:offset + val_len].decode('utf-8')
                offset += val_len
            else:
                logger.warning(f"Unknown EventStream header value type: {value_type} for header '{name}'")
                # Skip unknown types — we can't parse what we don't know
                break

        return headers
