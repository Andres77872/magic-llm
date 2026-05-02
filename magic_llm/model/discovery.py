"""Normalized discovery model contract for provider model catalog listing.

Per spec.md Section "Normalized Discovery Contract":
- Provides provider-agnostic model metadata structure
- Preserves external_id as-is from provider API
- Normalizes capabilities into boolean flags
- Preserves raw payload for debugging
- Pricing is optional — only some providers expose it
"""

from __future__ import annotations

from typing import Optional, Any, Dict
from pydantic import BaseModel, ConfigDict, Field


class ModelCapabilities(BaseModel):
    """Normalized capability flags for discovered models.
    
    Per spec.md Section "Capabilities (normalized boolean flags)":
    - All capabilities default to appropriate values
    - Chat defaults True (most models support chat)
    - Embedding/completion/vision/audio default False
    """
    
    model_config = ConfigDict(extra="forbid")
    
    chat: bool = Field(default=True, description="Supports chat/completion API")
    completion: bool = Field(default=False, description="Supports raw completion (legacy)")
    embedding: bool = Field(default=False, description="Supports embedding generation")
    vision: bool = Field(default=False, description="Supports image/vision inputs")
    audio_input: bool = Field(default=False, description="Supports audio input")
    function_calling: bool = Field(default=False, description="Supports tool/function calling")
    streaming: bool = Field(default=True, description="Supports streaming responses")
    reasoning: bool = Field(default=False, description="Supports extended reasoning/thinking")


class PricingInfo(BaseModel):
    """Optional pricing information from provider API.
    
    Per spec.md Section "Pricing (optional)":
    - Only some providers expose pricing (OpenRouter, Azure Foundry)
    - SambaNova pricing is optional — may return null
    - Prices are per 1M tokens
    """
    
    input_per_million: Optional[float] = Field(default=None, description="Price per 1M input/prompt tokens")
    output_per_million: Optional[float] = Field(default=None, description="Price per 1M output/completion tokens")


class NormalizedDiscoveredModel(BaseModel):
    """Provider-normalized model discovery result.
    
    Per spec.md Section "Normalized Discovery Contract":
    - external_id: Provider's raw model ID (preserved as-is)
    - provider: Engine identifier (openai, anthropic, etc.)
    - capabilities: Normalized boolean flags
    - pricing: Optional — only some providers expose
    - raw: Original provider response preserved for debugging
    
    This contract enables:
    - Single UI component for model preview
    - Admin filtering by capability
    - Import into any provider via existing import_model() contract
    """
    
    model_config = ConfigDict(extra="forbid")
    
    # Identity (required)
    external_id: str = Field(..., description="Provider's raw model ID (preserved as-is)")
    provider: str = Field(..., description="Engine identifier (openai, anthropic, google, etc.)")
    
    # Display (optional, provider-specific)
    display_name: Optional[str] = Field(default=None, description="Human-readable model name")
    description: Optional[str] = Field(default=None, description="Model description from provider")
    
    # Capabilities (normalized boolean flags)
    capabilities: ModelCapabilities = Field(
        default_factory=ModelCapabilities,
        description="Normalized capability flags"
    )
    
    # Limits (optional)
    context_window: Optional[int] = Field(default=None, description="Maximum context window tokens")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum output/completion tokens")
    
    # Pricing (optional — only some providers expose)
    pricing: Optional[PricingInfo] = Field(
        default=None,
        description="Optional pricing info from provider API"
    )
    
    # Raw payload for debugging
    raw: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original provider API response preserved"
    )