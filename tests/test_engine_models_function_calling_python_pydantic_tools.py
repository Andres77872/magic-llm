"""Live contract: callable and Pydantic tools work at init and call time."""
import pytest
from pydantic import BaseModel

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from conftest import assert_forced_weather_response, assert_forced_weather_stream, get_provider_key


pytestmark = pytest.mark.provider_functional


def get_weather(location: str):
    """Retrieve the current weather for a city."""


class GetForecast(BaseModel):
    """Forecast for a given location and days."""
    location: str
    days: int


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-3-haiku-20240307"),
    ("deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
])
@pytest.mark.parametrize("mode", ["init", "override", "stream"])
def test_forced_callable_with_pydantic_tools(provider_keys, provider, model, mode):
    chat = ModelChat()
    chat.add_user_message("Please check the weather for Bogotá.")
    options = {"tools": [get_weather, GetForecast],
               "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
    defaults = {"tools": [GetForecast], "tool_choice": "auto"} if mode == "override" else options
    keys = get_provider_key(provider_keys, provider, provider)
    client = MagicLLM(model=model, **keys, **defaults)
    if mode == "stream":
        assert_forced_weather_stream(client.llm.stream_generate(chat), "get_weather")
    else:
        response = client.llm.generate(chat, **(options if mode == "override" else {}))
        assert_forced_weather_response(response, "get_weather")
