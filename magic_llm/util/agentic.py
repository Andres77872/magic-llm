import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

from magic_llm.model.ModelChatResponse import ModelChatResponse, ToolCall, FunctionCall
from magic_llm.model.ModelChatStream import ChatCompletionModel, DeltaModel, ChoiceModel

if TYPE_CHECKING:
    from magic_llm.model.ModelChat import ModelChat

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
    content_separator: str = "\n\n",
    **kwargs: Any,
) -> ModelChatResponse:
    """
    ReAct-style agentic loop that invokes tools as directed by the model until a final response.

    Implements the ReAct (Reasoning and Acting) pattern where:
    - The LLM reasons about the task and decides on actions (tool calls)
    - Tools are executed and observations are fed back to the LLM
    - The loop continues until the LLM produces a response without tool calls
    - All LLM content/reasoning is concatenated into the final response

    Supports patterns like: llm-tool-llm-tool-tool-llm where each llm response
    content is preserved and concatenated.

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
        content_separator: Separator used to join content from multiple LLM responses
        **kwargs: Extra args passed through to the underlying engine .generate()

    Returns:
        Final ModelChatResponse with concatenated content from all LLM responses.
    """
    from magic_llm.model.ModelChat import ModelChat
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

    # Collect all LLM content throughout the ReAct loop
    collected_content: List[str] = []
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

        # Collect any content the model produced (reasoning/thoughts)
        if content:
            collected_content.append(content)
            chat.add_assistant_message(content)
            logger.debug(f"[agentic] collected content (iteration {iteration_index + 1}):\n" + _safe_preview(content, 1200))

        # If no tool calls, the loop is complete
        if not tool_calls or len(tool_calls) == 0:
            logger.info("[agentic] no tool calls in response, ending loop")
            break

        # Execute each tool call and collect observations
        tool_results: List[str] = []
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
            tool_results.append(feedback)

        # Add all tool results as a single user message (observation)
        if tool_results:
            combined_feedback = "\n\n".join(tool_results)
            chat.add_user_message(combined_feedback)
            logger.debug("[agentic] appended tool feedback to conversation")

    # Handle edge case: no responses during loop
    if last_response is None:
        last_response = client.llm.generate(chat, tools=tools, tool_choice=tool_choice, model=model, **kwargs)
        if last_response.content:
            collected_content.append(last_response.content)
        logger.warning("[agentic] no responses during loop; produced a final response as fallback")

    # Build final response with concatenated content
    final_content = content_separator.join(collected_content) if collected_content else None
    logger.info(f"[agentic] loop completed with {len(collected_content)} content segments")

    # Update the last response content with concatenated content
    if final_content and last_response.choices:
        last_response.choices[0].message.content = final_content

    return last_response


def _accumulate_tool_calls_from_delta(
    accumulated: Dict[int, Dict[str, Any]],
    delta_tool_calls: Optional[List[Any]]
) -> None:
    """Accumulate streaming tool_calls deltas into complete tool calls."""
    if not delta_tool_calls:
        return

    for tc in delta_tool_calls:
        idx = tc.index if hasattr(tc, 'index') and tc.index is not None else 0
        if idx not in accumulated:
            accumulated[idx] = {
                'id': '',
                'type': 'function',
                'function': {'name': '', 'arguments': ''}
            }

        if tc.id:
            accumulated[idx]['id'] = tc.id
        if tc.function:
            if tc.function.name:
                accumulated[idx]['function']['name'] = tc.function.name
            if tc.function.arguments:
                accumulated[idx]['function']['arguments'] += tc.function.arguments


def _build_tool_calls_from_accumulated(
    accumulated: Dict[int, Dict[str, Any]]
) -> List[ToolCall]:
    """Convert accumulated tool call data to ToolCall objects."""
    result = []
    for idx in sorted(accumulated.keys()):
        data = accumulated[idx]
        result.append(ToolCall(
            id=data['id'],
            type=data['type'],
            function=FunctionCall(
                name=data['function']['name'],
                arguments=data['function']['arguments']
            )
        ))
    return result


def _create_separator_chunk(separator: str, model: str, chunk_id: str) -> ChatCompletionModel:
    """Create a synthetic chunk containing the separator content."""
    return ChatCompletionModel(
        id=chunk_id,
        model=model,
        choices=[ChoiceModel(delta=DeltaModel(content=separator))]
    )


def run_agentic_stream(
    client,
    user_input: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[ToolSpec]] = None,
    tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    max_iterations: int = 8,
    model: Optional[str] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    extra_messages: Optional[List[Dict[str, Any]]] = None,
    content_separator: str = "\n\n",
    **kwargs: Any,
) -> Iterator[ChatCompletionModel]:
    """
    Streaming ReAct-style agentic loop that yields chunks while executing tools.

    Implements the ReAct pattern with streaming output:
    - Yields ChatCompletionModel chunks as they arrive from the LLM
    - Accumulates content and tool_calls during streaming
    - Executes tools when a response completes with tool_calls
    - Continues the loop until no more tool calls
    - All content is concatenated across iterations

    Args:
        client: MagicLLM client instance
        user_input: User's initial prompt
        system_prompt: Optional system message
        tools: Tools to advertise to the model
        tool_functions: Mapping of tool name -> callable
        max_iterations: Safety cap for tool loops
        model: Optional model override
        tool_choice: Tool choice directive
        extra_messages: Optional pre-existing messages
        content_separator: Separator for joining content segments
        **kwargs: Extra args passed to stream_generate

    Yields:
        ChatCompletionModel chunks from the streaming response
    """
    from magic_llm.model.ModelChat import ModelChat
    chat = ModelChat(system=system_prompt)
    for msg in extra_messages or []:
        chat.add_message(role=msg["role"], content=msg["content"])
    chat.add_user_message(user_input)

    registry = _build_tool_registry(tools, tool_functions)
    logger.info("[agentic_stream] Starting streaming agentic run")

    collected_content: List[str] = []

    for iteration_index in range(max_iterations):
        logger.info(f"[agentic_stream] iteration {iteration_index + 1}/{max_iterations}")

        iteration_content = ''
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        last_chunk = None

        for chunk in client.llm.stream_generate(chat, tools=tools, tool_choice=tool_choice, **kwargs):
            last_chunk = chunk
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta:
                if delta.content:
                    iteration_content += delta.content
                if delta.tool_calls:
                    _accumulate_tool_calls_from_delta(accumulated_tool_calls, delta.tool_calls)

            yield chunk

        if iteration_content:
            collected_content.append(iteration_content)
            chat.add_assistant_message(iteration_content)
            logger.debug(f"[agentic_stream] collected content (iteration {iteration_index + 1})")

        tool_calls = _build_tool_calls_from_accumulated(accumulated_tool_calls) if accumulated_tool_calls else []

        if not tool_calls:
            logger.info("[agentic_stream] no tool calls, ending loop")
            break

        logger.info(f"[agentic_stream] executing {len(tool_calls)} tool(s)")
        tool_results: List[str] = []
        for call in tool_calls:
            if not call or not call.function or not call.function.name:
                continue

            name = call.function.name
            args_raw = call.function.arguments
            parsed_args, input_echo = _parse_tool_arguments(args_raw)

            fn = registry.get(name)
            if not fn:
                output = _coerce_tool_output({"error": f"Unknown tool: {name}"})
                logger.warning(f"[agentic_stream] tool '{name}' not found")
            else:
                try:
                    if isinstance(parsed_args, dict):
                        result = fn(**parsed_args)
                    elif parsed_args is None or parsed_args == "":
                        result = fn()
                    else:
                        result = fn(parsed_args)
                    output = _coerce_tool_output(result)
                    logger.debug(f"[agentic_stream] tool '{name}' completed")
                except Exception as e:
                    output = _coerce_tool_output({"error": str(e)})
                    logger.error(f"[agentic_stream] tool '{name}' error: {e}")

            feedback = _format_tool_feedback(name=name, input_str=input_echo, output_str=output)
            tool_results.append(feedback)

        if tool_results:
            combined_feedback = "\n\n".join(tool_results)
            chat.add_user_message(combined_feedback)

            if iteration_content and content_separator:
                separator_chunk = _create_separator_chunk(
                    separator=content_separator,
                    model=last_chunk.model if last_chunk else '',
                    chunk_id=last_chunk.id if last_chunk else 'separator'
                )
                yield separator_chunk

    logger.info(f"[agentic_stream] completed with {len(collected_content)} content segments")


async def run_agentic_async(
    client,
    user_input: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[ToolSpec]] = None,
    tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    max_iterations: int = 8,
    model: Optional[str] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    extra_messages: Optional[List[Dict[str, Any]]] = None,
    content_separator: str = "\n\n",
    **kwargs: Any,
) -> ModelChatResponse:
    """Async ReAct-style agentic loop using client.llm.async_generate().

    Mirrors run_agentic() but:
    1. Uses `await client.llm.async_generate(...)` instead of `client.llm.generate(...)`
    2. Supports async tool callables: `if asyncio.iscoroutinefunction(fn): result = await fn(**args)`
    3. Executes tools sequentially (parallel via asyncio.gather is a future enhancement)

    Args:
        client: MagicLLM client instance
        user_input: User's initial prompt
        system_prompt: Optional system message
        tools: Tools to advertise to the model (callables, dict tool specs, or pydantic)
        tool_functions: Optional mapping of tool name -> callable for execution
        max_iterations: Safety cap for tool loops
        model: Optional model override
        tool_choice: Tool choice directive (e.g., "auto", "none", "required")
        extra_messages: Optional list of pre-existing messages to seed the conversation
        content_separator: Separator used to join content from multiple LLM responses
        **kwargs: Extra args passed through to the underlying engine .async_generate()

    Returns:
        Final ModelChatResponse with concatenated content from all LLM responses.
    """
    from magic_llm.model.ModelChat import ModelChat

    chat = ModelChat(system=system_prompt)
    for msg in extra_messages or []:
        chat.add_message(role=msg["role"], content=msg["content"])
    chat.add_user_message(user_input)

    # Build registry of name -> callable for execution
    registry = _build_tool_registry(tools, tool_functions)
    logger.info("[agentic_async] Starting async agentic run")
    if system_prompt:
        logger.debug("[agentic_async] system_prompt:\n" + _safe_preview(system_prompt, 600))
    logger.debug("[agentic_async] user_input:\n" + _safe_preview(user_input, 600))
    logger.info(
        "[agentic_async] available tools: "
        + (", ".join(sorted(registry.keys())) if registry else "<none>")
    )

    # Collect all LLM content throughout the ReAct loop
    collected_content: List[str] = []
    last_response: Optional[ModelChatResponse] = None

    for iteration_index in range(max_iterations):
        logger.info(
            f"[agentic_async] iteration {iteration_index + 1}/{max_iterations}: requesting model response..."
        )
        response = await client.llm.async_generate(
            chat, tools=tools, tool_choice=tool_choice, **kwargs
        )
        last_response = response

        tool_calls: Optional[List[ToolCall]] = response.tool_calls
        content: Optional[str] = response.content
        finish_reason = getattr(response, "finish_reason", None)
        model_name = getattr(response, "model", None)
        num_tool_calls = len(tool_calls) if tool_calls else 0
        logger.info(
            "[agentic_async] model reply: "
            f"model={model_name} finish_reason={finish_reason} tool_calls={num_tool_calls} "
            f"content={'present' if content else 'none'}"
        )

        # Collect any content the model produced (reasoning/thoughts)
        if content:
            collected_content.append(content)
            chat.add_assistant_message(content)
            logger.debug(
                f"[agentic_async] collected content (iteration {iteration_index + 1}):\n"
                + _safe_preview(content, 1200)
            )

        # If no tool calls, the loop is complete
        if not tool_calls or len(tool_calls) == 0:
            logger.info("[agentic_async] no tool calls in response, ending loop")
            break

        # Execute each tool call and collect observations
        tool_results: List[str] = []
        for call in tool_calls:
            if not call or not call.function or not call.function.name:
                continue

            name = call.function.name
            args_raw = call.function.arguments
            parsed_args, input_echo = _parse_tool_arguments(args_raw)

            fn = registry.get(name)
            if not fn:
                output = _coerce_tool_output({"error": f"Unknown tool: {name}"})
                logger.warning(f"[agentic_async] tool '{name}' not found; skipping")
            else:
                try:
                    logger.debug(
                        f"[agentic_async] executing tool: {name} with args: "
                        + _safe_preview(parsed_args, 600)
                    )
                    # Detect async tool callables and await them
                    if asyncio.iscoroutinefunction(fn):
                        if isinstance(parsed_args, dict):
                            result = await fn(**parsed_args)
                        elif parsed_args is None or parsed_args == "":
                            result = await fn()
                        else:
                            result = await fn(parsed_args)
                    else:
                        if isinstance(parsed_args, dict):
                            result = fn(**parsed_args)
                        elif parsed_args is None or parsed_args == "":
                            result = fn()
                        else:
                            result = fn(parsed_args)
                    output = _coerce_tool_output(result)
                    logger.debug(
                        f"[agentic_async] tool '{name}' completed; output preview:\n"
                        + _safe_preview(output, 800)
                    )
                except Exception as e:
                    output = _coerce_tool_output({"error": str(e)})
                    logger.error(f"[agentic_async] tool '{name}' raised error: {e}")

            feedback = _format_tool_feedback(name=name, input_str=input_echo, output_str=output)
            tool_results.append(feedback)

        # Add all tool results as a single user message (observation)
        if tool_results:
            combined_feedback = "\n\n".join(tool_results)
            chat.add_user_message(combined_feedback)
            logger.debug("[agentic_async] appended tool feedback to conversation")

    # Handle edge case: no responses during loop
    if last_response is None:
        last_response = await client.llm.async_generate(
            chat, tools=tools, tool_choice=tool_choice, model=model, **kwargs
        )
        if last_response.content:
            collected_content.append(last_response.content)
        logger.warning(
            "[agentic_async] no responses during loop; produced a final response as fallback"
        )

    # Build final response with concatenated content
    final_content = content_separator.join(collected_content) if collected_content else None
    logger.info(f"[agentic_async] loop completed with {len(collected_content)} content segments")

    # Update the last response content with concatenated content
    if final_content and last_response.choices:
        last_response.choices[0].message.content = final_content

    return last_response


async def run_agentic_stream_async(
    client,
    user_input: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[ToolSpec]] = None,
    tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    max_iterations: int = 8,
    model: Optional[str] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    extra_messages: Optional[List[Dict[str, Any]]] = None,
    content_separator: str = "\n\n",
    **kwargs: Any,
) -> AsyncIterator[ChatCompletionModel]:
    """Async streaming ReAct-style agentic loop using async_stream_generate().

    Mirrors run_agentic_stream() but:
    1. Uses `async for chunk in client.llm.async_stream_generate(...)`
    2. Supports async tool callables via asyncio.iscoroutinefunction()
    3. Yields ChatCompletionModel chunks as they arrive

    Args:
        client: MagicLLM client instance
        user_input: User's initial prompt
        system_prompt: Optional system message
        tools: Tools to advertise to the model
        tool_functions: Mapping of tool name -> callable
        max_iterations: Safety cap for tool loops
        model: Optional model override
        tool_choice: Tool choice directive
        extra_messages: Optional pre-existing messages
        content_separator: Separator for joining content segments
        **kwargs: Extra args passed to async_stream_generate

    Yields:
        ChatCompletionModel chunks from the streaming response
    """
    from magic_llm.model.ModelChat import ModelChat

    chat = ModelChat(system=system_prompt)
    for msg in extra_messages or []:
        chat.add_message(role=msg["role"], content=msg["content"])
    chat.add_user_message(user_input)

    registry = _build_tool_registry(tools, tool_functions)
    logger.info("[agentic_stream_async] Starting async streaming agentic run")

    collected_content: List[str] = []

    for iteration_index in range(max_iterations):
        logger.info(
            f"[agentic_stream_async] iteration {iteration_index + 1}/{max_iterations}"
        )

        iteration_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        last_chunk = None

        async for chunk in client.llm.async_stream_generate(
            chat, tools=tools, tool_choice=tool_choice, **kwargs
        ):
            last_chunk = chunk
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta:
                if delta.content:
                    iteration_content += delta.content
                if delta.tool_calls:
                    _accumulate_tool_calls_from_delta(accumulated_tool_calls, delta.tool_calls)

            yield chunk

        if iteration_content:
            collected_content.append(iteration_content)
            chat.add_assistant_message(iteration_content)
            logger.debug(
                f"[agentic_stream_async] collected content (iteration {iteration_index + 1})"
            )

        tool_calls = (
            _build_tool_calls_from_accumulated(accumulated_tool_calls)
            if accumulated_tool_calls
            else []
        )

        if not tool_calls:
            logger.info("[agentic_stream_async] no tool calls, ending loop")
            break

        logger.info(f"[agentic_stream_async] executing {len(tool_calls)} tool(s)")
        tool_results: List[str] = []
        for call in tool_calls:
            if not call or not call.function or not call.function.name:
                continue

            name = call.function.name
            args_raw = call.function.arguments
            parsed_args, input_echo = _parse_tool_arguments(args_raw)

            fn = registry.get(name)
            if not fn:
                output = _coerce_tool_output({"error": f"Unknown tool: {name}"})
                logger.warning(f"[agentic_stream_async] tool '{name}' not found")
            else:
                try:
                    # Detect async tool callables and await them
                    if asyncio.iscoroutinefunction(fn):
                        if isinstance(parsed_args, dict):
                            result = await fn(**parsed_args)
                        elif parsed_args is None or parsed_args == "":
                            result = await fn()
                        else:
                            result = await fn(parsed_args)
                    else:
                        if isinstance(parsed_args, dict):
                            result = fn(**parsed_args)
                        elif parsed_args is None or parsed_args == "":
                            result = fn()
                        else:
                            result = fn(parsed_args)
                    output = _coerce_tool_output(result)
                    logger.debug(f"[agentic_stream_async] tool '{name}' completed")
                except Exception as e:
                    output = _coerce_tool_output({"error": str(e)})
                    logger.error(f"[agentic_stream_async] tool '{name}' error: {e}")

            feedback = _format_tool_feedback(name=name, input_str=input_echo, output_str=output)
            tool_results.append(feedback)

        if tool_results:
            combined_feedback = "\n\n".join(tool_results)
            chat.add_user_message(combined_feedback)

            if iteration_content and content_separator:
                separator_chunk = _create_separator_chunk(
                    separator=content_separator,
                    model=last_chunk.model if last_chunk else "",
                    chunk_id=last_chunk.id if last_chunk else "separator",
                )
                yield separator_chunk

    logger.info(
        f"[agentic_stream_async] completed with {len(collected_content)} content segments"
    )
