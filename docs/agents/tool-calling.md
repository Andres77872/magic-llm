# Tool calling

Magic LLM normalizes tool definitions into provider-specific schemas. You can pass Python callables, OpenAI-style JSON specs, or Pydantic models.

## Python callables

```python
def search_docs(query: str, limit: int = 5) -> list[str]:
    """Search project documentation."""
    return [f"Result for {query}"][:limit]

response = client.run_agent(
    user_input="Search the docs for model discovery.",
    tools=[search_docs],
)
```

The function name becomes the tool name, the docstring becomes the description, and type hints become the argument schema.

## OpenAI-style JSON specs

```python
tool_specs = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}]

def add(a: int, b: int) -> int:
    return a + b

response = client.run_agent(
    user_input="Add 7 and 35.",
    tools=tool_specs,
    tool_functions={"add": add},
)
```

Use this shape when you want exact control over the tool schema.

## Pydantic model tools

```python
from pydantic import BaseModel, Field

class WeatherRequest(BaseModel):
    """Get weather for a city."""
    city: str = Field(..., description="City name")

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"

response = client.run_agent(
    user_input="Check the weather in Montevideo.",
    tools=[WeatherRequest],
    tool_functions={"WeatherRequest": get_weather},
)
```

Pydantic classes provide schemas. Execution still needs a callable path, so map the model class name to a callable with `tool_functions`. If you are unsure, prefer plain callables or JSON specs with `tool_functions`.

## Unsupported engines

Cohere and Cloudflare call the tooling guard and reject tools. Use direct chat or switch to a tool-capable provider.
