"""Pattern tables for capability inference.

All regex patterns and context window maps are defined here as
module-level constants so they can be imported by strategies,
tests, and snapshot-verified independently.
"""

# ── Vision capability patterns ────────────────────────────────────────
# Matched against the model ID (case-insensitive).
# Covers OpenAI GPT-4o/Turbo/Vision, Claude 3, Gemini, and any model
# with "vision" in the name.
VISION_PATTERNS = [
    r"gpt-4o",  # GPT-4o models have vision
    r"gpt-4-turbo",  # GPT-4 Turbo with vision
    r"gpt-4-vision",  # Explicit vision models
    r"claude-3",  # Claude 3 models (if via OpenAI-compatible)
    r"vision",  # Any model with 'vision' in name
    r"gemini",  # Gemini models (if via OpenAI-compatible)
]

# ── Embedding capability patterns ─────────────────────────────────────
EMBEDDING_PATTERNS = [
    r"embed",
    r"text-embedding",
    r"embedding",
]

# ── Function-calling capability patterns ──────────────────────────────
FUNCTION_CALLING_PATTERNS = [
    r"gpt-4",
    r"gpt-3.5-turbo",
    r"claude",
]

# ── Verified media capability patterns ─────────────────────────────────
# Conservative by design: these patterns only mark media flags when the
# runtime has a supported path in the media matrix. No image-output patterns
# are added in this change.
AUDIO_INPUT_PATTERNS = [
    r"whisper",
]

AUDIO_OUTPUT_PATTERNS = [
    r"tts",
    r"sonic",
    r"text-to-speech",
]

# ── Context window lookups (model → max context tokens) ───────────────
# Keys are regexes searched (case-insensitively) against the model ID.
# Order matters: the first match wins, so more specific families MUST come
# before their prefixes ("gpt-4o" before "gpt-4", "gpt-4.1" before "gpt-4").
# Only used as a fallback when the provider listing carries no limit.
CONTEXT_WINDOW_MAP = {
    r"gpt-5": 400000,
    r"gpt-4\.1": 1047576,
    r"gpt-4o-mini": 128000,
    r"gpt-4o": 128000,
    r"gpt-4-turbo": 128000,
    r"gpt-4-32k": 32768,
    r"gpt-4(?![\w.])": 8192,
    r"gpt-3\.5-turbo-16k": 16385,
    r"gpt-3\.5-turbo": 16385,
    r"(?<![\w.])o[34](-mini)?(?![\w.])": 200000,
    r"(?<![\w.])o1-(mini|preview)": 128000,
    r"(?<![\w.])o1(?![\w.])": 200000,
}
