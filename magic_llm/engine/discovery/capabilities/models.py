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
    r"gpt-4o",           # GPT-4o models have vision
    r"gpt-4-turbo",      # GPT-4 Turbo with vision
    r"gpt-4-vision",     # Explicit vision models
    r"claude-3",         # Claude 3 models (if via OpenAI-compatible)
    r"vision",           # Any model with 'vision' in name
    r"gemini",           # Gemini models (if via OpenAI-compatible)
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

# ── Context window lookups (model → max context tokens) ───────────────
CONTEXT_WINDOW_MAP = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-preview": 128000,
}
