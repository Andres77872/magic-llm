import mimetypes
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


DEFAULT_MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS = frozenset({
    'json',
    'verbose_json',
    'text',
    'srt',
    'vtt',
})
SUPPORTED_AUDIO_CONTENT_TYPES = frozenset({
    'audio/wav',
    'audio/x-wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/ogg',
    'audio/flac',
    'audio/mp4',
    'video/mp4',
})
_CONTENT_TYPE_EXTENSIONS = {
    'audio/wav': 'wav',
    'audio/x-wav': 'wav',
    'audio/mpeg': 'mp3',
    'audio/mp3': 'mp3',
    'audio/ogg': 'ogg',
    'audio/flac': 'flac',
    'audio/mp4': 'm4a',
    'video/mp4': 'mp4',
}
_EXTENSION_CONTENT_TYPES = {
    'wav': 'audio/wav',
    'wave': 'audio/wav',
    'mp3': 'audio/mpeg',
    'mpeg': 'audio/mpeg',
    'mpga': 'audio/mpeg',
    'ogg': 'audio/ogg',
    'oga': 'audio/ogg',
    'flac': 'audio/flac',
    'm4a': 'audio/mp4',
    'mp4': 'audio/mp4',
}


@dataclass(frozen=True)
class AudioUploadMetadata:
    filename: str
    content_type: str


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return content_type.strip().lower()


def _filename_extension(filename: str | None) -> str | None:
    if not filename or '.' not in filename:
        return None
    return filename.rsplit('.', 1)[-1].lower()


def _guess_content_type_from_filename(filename: str | None) -> str | None:
    extension = _filename_extension(filename)
    if extension in _EXTENSION_CONTENT_TYPES:
        return _EXTENSION_CONTENT_TYPES[extension]
    guessed, _ = mimetypes.guess_type(filename or '')
    return _normalize_content_type(guessed)


def _guess_filename_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    extension = _CONTENT_TYPE_EXTENSIONS.get(content_type)
    return f'audio.{extension}' if extension else None


def _infer_audio_metadata_from_bytes(file_bytes: bytes) -> AudioUploadMetadata | None:
    if file_bytes.startswith(b'RIFF') and file_bytes[8:12] == b'WAVE':
        return AudioUploadMetadata(filename='audio.wav', content_type='audio/wav')
    if file_bytes.startswith(b'ID3') or (len(file_bytes) > 1 and file_bytes[0] == 0xFF and (file_bytes[1] & 0xE0) == 0xE0):
        return AudioUploadMetadata(filename='audio.mp3', content_type='audio/mpeg')
    if file_bytes.startswith(b'OggS'):
        return AudioUploadMetadata(filename='audio.ogg', content_type='audio/ogg')
    if file_bytes.startswith(b'fLaC'):
        return AudioUploadMetadata(filename='audio.flac', content_type='audio/flac')
    if len(file_bytes) >= 12 and file_bytes[4:8] == b'ftyp':
        major = file_bytes[8:12].lower()
        if major in {b'm4a ', b'm4b ', b'mp42', b'isom', b'mp41'}:
            return AudioUploadMetadata(filename='audio.m4a', content_type='audio/mp4')
    return None


class AudioSpeechRequest(BaseModel):
    input: str
    model: str
    voice: str
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1


class AudioTranscriptionsRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    file: bytes
    model: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = 'json'
    temperature: Optional[float] = 0
    filename: Optional[str] = None
    content_type: Optional[str] = None

    @field_validator('file')
    @classmethod
    def validate_file(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError('Audio transcription file must be non-empty bytes')
        if len(value) > DEFAULT_MAX_AUDIO_UPLOAD_BYTES:
            raise ValueError(
                f'Audio transcription file exceeds maximum size of {DEFAULT_MAX_AUDIO_UPLOAD_BYTES} bytes'
            )
        return value

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        filename = value.strip()
        if not filename:
            raise ValueError('Audio transcription filename must be non-empty when provided')
        if '/' in filename or '\\' in filename or filename in {'.', '..'}:
            raise ValueError('Audio transcription filename must be a plain filename, not a path')
        return filename

    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, value: Optional[str]) -> Optional[str]:
        content_type = _normalize_content_type(value)
        if content_type is None:
            return None
        if not re.fullmatch(r'[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+', content_type):
            raise ValueError('Audio transcription content_type must use MIME type/subtype format')
        return content_type

    @field_validator('response_format')
    @classmethod
    def validate_response_format(cls, value: Optional[str]) -> str:
        response_format = (value or 'json').strip().lower()
        if response_format not in SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS:
            supported = ', '.join(sorted(SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS))
            raise ValueError(f'Unsupported transcription response_format {value!r}. Supported: {supported}')
        return response_format

    def resolve_upload_metadata(self, *, require_metadata: bool = True) -> AudioUploadMetadata:
        """Resolve accurate filename/content-type metadata for provider uploads.

        Caller-supplied metadata wins when present. Missing metadata is inferred
        from the filename/content type or from conservative audio byte signatures.
        Unknown bytes do not default to fake MP3/WAV metadata.
        """
        filename = self.filename
        content_type = _normalize_content_type(self.content_type)
        inferred = _infer_audio_metadata_from_bytes(self.file)

        def _reject_if_filename_conflicts_with_content_type() -> None:
            extension_content_type = _guess_content_type_from_filename(filename)
            if filename and extension_content_type and content_type and extension_content_type != content_type:
                raise ValueError(
                    f'Audio filename extension implies {extension_content_type}, '
                    f'not validated content_type {content_type}'
                )

        def _reject_if_bytes_conflict_with_content_type() -> None:
            if inferred and content_type and inferred.content_type != content_type:
                raise ValueError(
                    f'Audio bytes appear to be {inferred.content_type}, not caller-provided {content_type}'
                )

        if filename and content_type:
            _reject_if_bytes_conflict_with_content_type()
            _reject_if_filename_conflicts_with_content_type()
            return AudioUploadMetadata(filename=filename, content_type=content_type)

        if filename and not content_type:
            content_type = _guess_content_type_from_filename(filename)
            _reject_if_bytes_conflict_with_content_type()

        if content_type and not filename:
            filename = _guess_filename_from_content_type(content_type)
            _reject_if_bytes_conflict_with_content_type()

        if not filename or not content_type:
            if inferred:
                filename = filename or inferred.filename
                content_type = content_type or inferred.content_type

        if filename and content_type:
            _reject_if_filename_conflicts_with_content_type()
            _reject_if_bytes_conflict_with_content_type()
            return AudioUploadMetadata(filename=filename, content_type=content_type)

        if require_metadata:
            raise ValueError(
                'Unable to infer audio filename/content_type safely; provide filename and content_type explicitly'
            )
        return AudioUploadMetadata(
            filename=filename or 'audio.bin',
            content_type=content_type or 'application/octet-stream',
        )

    def transcription_fields(self, *, model: Optional[str] = None) -> dict[str, str]:
        """Return OpenAI-compatible non-file multipart STT fields."""
        selected_model = self.model or model
        if not selected_model:
            raise ValueError('Audio transcription request requires a model')
        fields = {'model': selected_model}
        if self.language:
            fields['language'] = self.language
        if self.prompt:
            fields['prompt'] = self.prompt
        if self.response_format:
            fields['response_format'] = self.response_format
        if self.temperature is not None:
            fields['temperature'] = str(self.temperature)
        return fields

    @property
    def is_json_response_format(self) -> bool:
        return self.response_format in {'json', 'verbose_json'}
