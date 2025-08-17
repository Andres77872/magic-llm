import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from magic_llm.model.ModelChatResponse import ModelChatResponse, ToolCall


ToolSpec = Union[Callable[..., Any], Dict[str, Any], Any]

logger = logging.getLogger(__name__)

def _coerce_tool_output(value: Any) -> str:
    """Always return a string representation of a tool's output."""
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)
    except Exception:
        return str(value)


def _safe_preview(value: Any, max_len: int = 800) -> str:
    """Return a printable, truncated preview for any value."""
    try:
        if isinstance(value, str):
            s = value
        else:
            s = json.dumps(value)
    except Exception:
        s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"... [truncated {len(s) - max_len} chars]"
    return s


def _build_tool_registry(
    tools: Optional[List[ToolSpec]],
    tool_functions: Optional[Dict[str, Callable[..., Any]]],
) -> Dict[str, Callable[..., Any]]:
    """Create a name->callable registry from provided tools and explicit function map."""
    registry: Dict[str, Callable[..., Any]] = {}

    # Callables included directly in tools
    for tool in tools or []:
        if callable(tool) and not isinstance(tool, dict):
            name = getattr(tool, "__name__", None)
            if name:
                registry[name] = tool

        # If tool is a dict definition, we can only call it if user supplied a callable for its name
        elif isinstance(tool, dict):
            # {"type":"function","function":{"name":...}} or legacy {"name":...}
            fn_def = tool.get("function") if tool.get("type") == "function" else tool
            name = fn_def.get("name") if isinstance(fn_def, dict) else None
            if name and tool_functions and callable(tool_functions.get(name)):
                registry[name] = tool_functions[name]

    # Merge/override with explicit mappings
    for name, fn in (tool_functions or {}).items():
        if callable(fn):
            registry[name] = fn

    return registry


def _parse_tool_arguments(arg_str: Optional[str]) -> Tuple[Union[dict, str, None], str]:
    """Parse OpenAI-style JSON argument string; return (parsed, echo_string)."""
    if arg_str is None:
        return None, ""
    try:
        parsed = json.loads(arg_str)
        return parsed, json.dumps(parsed)
    except Exception:
        # Not JSON; return raw
        return arg_str, str(arg_str)


def _format_tool_feedback(name: str, input_str: str, output_str: str) -> str:
    return (
        f"tool: {name}\n"
        f"tool input: {input_str}\n"
        f"tool output: {output_str}"
    )


def run_agentic(
    client,
    user_input: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[ToolSpec]] = None,
    tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    max_iterations: int = 8,
    model: Optional[str] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    extra_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ModelChatResponse:
    """
    Basic agentic loop that invokes tools as directed by the model until a normal response is produced.

    - Builds a conversation with optional `system_prompt` and the initial `user_input`.
    - Passes `tools` to the model on each turn.
    - If the response contains tool_calls, executes the matching Python callables and appends
      a new user message with:
        tool: <name>\n
        tool input: <json or text>\n
        tool output: <string or json>
    - Repeats until the model returns normal content (no tool_calls).

    Args:
        client: MagicLLM client instance
        user_input: User's initial prompt
        system_prompt: Optional system message
        tools: Tools to advertise to the model (callables, dict tool specs, or pydantic)
        tool_functions: Optional mapping of tool name -> callable for execution
        max_iterations: Safety cap for tool loops
        model: Optional model override
        tool_choice: Tool choice directive (e.g., "auto", "none", "required", or {"name": ...})
        extra_messages: Optional list of pre-existing messages to seed the conversation
        **kwargs: Extra args passed through to the underlying engine .generate()

    Returns:
        Final ModelChatResponse with normal content when the loop ends.
    """

    from magic_llm.model.ModelChat import ModelChat
    # Initialize conversation
    chat = ModelChat(system=system_prompt)
    for msg in extra_messages or []:
        chat.add_message(role=msg["role"], content=msg["content"])  # type: ignore[index]
    chat.add_user_message(user_input)

    # Build registry of name -> callable for execution
    registry = _build_tool_registry(tools, tool_functions)
    logger.info("[agentic] Starting agentic run")
    if system_prompt:
        logger.debug("[agentic] system_prompt:\n" + _safe_preview(system_prompt, 600))
    logger.debug("[agentic] user_input:\n" + _safe_preview(user_input, 600))
    logger.info(
        "[agentic] available tools: "
        + (", ".join(sorted(registry.keys())) if registry else "<none>")
    )

    # Iterate until normal response or iteration cap
    last_response: Optional[ModelChatResponse] = None
    for iteration_index in range(max_iterations):
        logger.info(
            f"[agentic] iteration {iteration_index + 1}/{max_iterations}: requesting model response..."
        )
        response = client.llm.generate(chat, tools=tools, tool_choice=tool_choice, **kwargs)
        last_response = response

        tool_calls: Optional[List[ToolCall]] = response.tool_calls
        content: Optional[str] = response.content
        finish_reason = getattr(response, "finish_reason", None)
        model_name = getattr(response, "model", None)
        num_tool_calls = len(tool_calls) if tool_calls else 0
        logger.info(
            "[agentic] model reply: "
            f"model={model_name} finish_reason={finish_reason} tool_calls={num_tool_calls} "
            f"content={'present' if content else 'none'}"
        )

        # If model responded with normal content (no tool calls), we are done
        if (not tool_calls) or len(tool_calls) == 0:
            if content:
                # Preserve assistant reply in transcript for completeness
                chat.add_assistant_message(content)
                logger.debug("[agentic] final assistant content:\n" + _safe_preview(content, 1200))
            break

        # Execute each tool call; append results as a new user message
        for call in tool_calls:
            if not call or not call.function or not call.function.name:
                continue

            name = call.function.name
            args_raw = call.function.arguments
            parsed_args, input_echo = _parse_tool_arguments(args_raw)

            fn = registry.get(name)
            if not fn:
                output = _coerce_tool_output({"error": f"Unknown tool: {name}"})
                logger.warning(f"[agentic] tool '{name}' not found; skipping")
            else:
                try:
                    logger.debug(
                        f"[agentic] executing tool: {name} with args: "
                        + _safe_preview(parsed_args, 600)
                    )
                    # Support dict- or str-style arguments
                    if isinstance(parsed_args, dict):
                        result = fn(**parsed_args)
                    elif parsed_args is None or parsed_args == "":
                        result = fn()
                    else:
                        result = fn(parsed_args)
                    output = _coerce_tool_output(result)
                    logger.debug(
                        f"[agentic] tool '{name}' completed; output preview:\n"
                        + _safe_preview(output, 800)
                    )
                except Exception as e:
                    output = _coerce_tool_output({"error": str(e)})
                    logger.error(f"[agentic] tool '{name}' raised error: {e}")

            feedback = _format_tool_feedback(name=name, input_str=input_echo, output_str=output)
            chat.add_user_message(feedback)
            logger.debug("[agentic] appended tool feedback to conversation")

    # Fallback if loop ends without a response (should not happen normally)
    if last_response is None:
        last_response = client.llm.generate(chat, tools=tools, tool_choice=tool_choice, model=model, **kwargs)
        logger.warning("[agentic] no responses during loop; produced a final response as fallback")

    return last_response


