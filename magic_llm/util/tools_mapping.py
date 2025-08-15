import copy
from typing import Any, Dict, List, Optional, Tuple, Union

# Types
OpenAITool = Dict[str, Any]
ToolChoice = Union[str, Dict[str, Any], None]


def normalize_openai_tools(tools: Optional[List[OpenAITool]]) -> List[Dict[str, Any]]:
    """
    Normalize OpenAI-style tools to a canonical list of function definitions.

    Input may be any of:
    - [{"type": "function", "function": {name, description?, parameters?}}]
    - [{name, description?, parameters?}] (legacy)

    Returns a list of dicts with keys: name, description, parameters
    """
    if not tools:
        return []

    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if tool is None:
            continue
        # If wrapped in {type:function, function:{...}}
        if tool.get("type") == "function":
            fn_def = copy.deepcopy(tool.get("function", {}))
        else:
            # Legacy: tool itself is the function definition
            fn_def = copy.deepcopy(tool)

        name = fn_def.get("name")
        if not name:
            # skip invalid entries
            continue

        normalized.append({
            "name": name,
            "description": fn_def.get("description", ""),
            "parameters": fn_def.get("parameters", {
                "type": "object",
                "properties": {},
                "required": []
            })
        })
    return normalized


def normalize_openai_tool_choice(tool_choice: ToolChoice) -> ToolChoice:
    """
    Normalize tool_choice into one of:
    - 'auto' | 'none' | 'required'
    - {"name": <function_name>}  (canonical named-function directive)
    - None

    Accepts legacy forms like:
    - {"type": "function", "function": {"name": "..."}}
    - {"name": "..."}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        # Unknown string – pass through unchanged
        return tool_choice
    if isinstance(tool_choice, dict):
        # New-style OpenAI
        if tool_choice.get("type") == "function":
            name = tool_choice.get("function", {}).get("name")
            return {"name": name} if name else None
        # Legacy
        if "name" in tool_choice:
            return {"name": tool_choice["name"]}
    return None


def map_to_anthropic(tools: Optional[List[OpenAITool]], tool_choice: ToolChoice) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Map OpenAI-style tools/tool_choice into Anthropic schema.

    Returns (anthropic_tools, anthropic_tool_choice)
    """
    normalized_tools = normalize_openai_tools(tools)
    normalized_choice = normalize_openai_tool_choice(tool_choice)

    anthropic_tools: Optional[List[Dict[str, Any]]] = None
    if normalized_tools:
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            for t in normalized_tools
        ]

    anthropic_choice: Optional[Dict[str, Any]] = None
    if isinstance(normalized_choice, str):
        if normalized_choice == "auto":
            anthropic_choice = {"type": "auto"}
        elif normalized_choice == "required":
            anthropic_choice = {"type": "any"}
        elif normalized_choice == "none":
            anthropic_choice = None  # Anthropic has no explicit "none"
    elif isinstance(normalized_choice, dict) and normalized_choice.get("name"):
        anthropic_choice = {"type": "tool", "name": normalized_choice["name"]}

    return anthropic_tools, anthropic_choice


def map_to_openai(tools: Optional[List[OpenAITool]], tool_choice: ToolChoice) -> Tuple[Optional[List[Dict[str, Any]]], ToolChoice]:
    """
    Map normalized tools/tool_choice back to OpenAI-style request body.

    Returns (openai_tools, openai_tool_choice)
    """
    normalized_tools = normalize_openai_tools(tools)
    normalized_choice = normalize_openai_tool_choice(tool_choice)

    openai_tools: Optional[List[Dict[str, Any]]] = None
    if normalized_tools:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            for t in normalized_tools
        ]

    openai_choice: ToolChoice = None
    if isinstance(normalized_choice, str):
        openai_choice = normalized_choice
    elif isinstance(normalized_choice, dict) and normalized_choice.get("name"):
        openai_choice = {"type": "function", "function": {"name": normalized_choice["name"]}}

    return openai_tools, openai_choice


def coerce_tool_choice_to_string(tool_choice: ToolChoice, default: str = "auto") -> Optional[str]:
    """
    For providers that only accept string tool_choice, convert dict/directives to a string.
    If the input is a dict with a specific function, we downgrade to `default`.
    """
    normalized = normalize_openai_tool_choice(tool_choice)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return normalized
    # dict -> downgrade
    return default
