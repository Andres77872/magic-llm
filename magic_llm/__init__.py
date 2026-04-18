import logging
import warnings
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from magic_llm.base import MagicLlmBase

if TYPE_CHECKING:
    from magic_llm.model.ModelChatResponse import ModelChatResponse
    from magic_llm.model.ModelChatStream import ChatCompletionModel
    from magic_llm.util.agentic import ToolSpec
    from magic_llm.agent.types import AgentBudget
    from magic_llm.agent.hooks import AgentHooks

__version__ = '0.1.28-dev1'

logger = logging.getLogger(__name__)


class MagicLLM(MagicLlmBase):

    def __init__(self,
                 engine: str,
                 model: str | None = None,
                 private_key: str | None = None,
                 callback: Optional[Callable] = None,
                 **kwargs):
        super().__init__(engine=engine,
                         model=model,
                         private_key=private_key,
                         callback=callback,
                         **kwargs)

    def download_embedding_search_model(self):
        """
        Download the embedding search model.

        This method is intended for downloading models used for embedding-based search.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_embedding_search_model method is not yet implemented")
        raise NotImplementedError("The embedding search model download functionality is not yet implemented")

    def download_tagger_model(self):
        """
        Download the tagger model.

        This method is intended for downloading models used for tagging content.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_tagger_model method is not yet implemented")
        raise NotImplementedError("The tagger model download functionality is not yet implemented")

    def download_tags_dictionary(self):
        """
        Download the tags dictionary.

        This method is intended for downloading dictionaries used for tag mapping.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_tags_dictionary method is not yet implemented")
        raise NotImplementedError("The tags dictionary download functionality is not yet implemented")

    # ─── Legacy agentic methods (deprecated) ────────────────────────────────

    async def agentic(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        max_iterations: int = 8,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        **kwargs: Any,
    ) -> "ModelChatResponse":
        """Run the async agentic loop with this client.

        .. deprecated::
            Use :meth:`run_agent_async` instead. This legacy method delegates to
            the procedural ``run_agentic_async()`` and will be removed in a future
            version. The newer ``AsyncAgentLoop`` provides budget enforcement,
            lifecycle hooks, parallel tool execution, and structured state.

        Args:
            user_input: User's initial prompt
            system_prompt: Optional system message
            tools: Tools to advertise to the model
            tool_functions: Mapping of tool name -> callable
            max_iterations: Safety cap for tool loops
            model: Optional model override
            tool_choice: Tool choice directive
            extra_messages: Optional pre-existing messages
            content_separator: Separator for joining content segments
            **kwargs: Extra args passed to async_generate

        Returns:
            Final ModelChatResponse with concatenated content.
        """
        warnings.warn(
            "MagicLLM.agentic() is deprecated. Use MagicLLM.run_agent_async() instead. "
            "See docs for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        from magic_llm.util.agentic import run_agentic_async

        return await run_agentic_async(
            self,
            user_input=user_input,
            system_prompt=system_prompt,
            tools=tools,
            tool_functions=tool_functions,
            max_iterations=max_iterations,
            model=model,
            tool_choice=tool_choice,
            extra_messages=extra_messages,
            content_separator=content_separator,
            **kwargs,
        )

    async def agentic_stream(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        max_iterations: int = 8,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        **kwargs: Any,
    ) -> AsyncIterator["ChatCompletionModel"]:
        """Run the async streaming agentic loop with this client.

        .. deprecated::
            Use :meth:`run_agent_stream_async` instead. This legacy method delegates
            to the procedural ``run_agentic_stream_async()`` and will be removed in a
            future version. The newer ``AsyncAgentLoop`` provides budget enforcement,
            lifecycle hooks, parallel tool execution, and structured state.

        Args:
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
            ChatCompletionModel chunks from the streaming response.
        """
        warnings.warn(
            "MagicLLM.agentic_stream() is deprecated. Use MagicLLM.run_agent_stream_async() instead. "
            "See docs for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        from magic_llm.util.agentic import run_agentic_stream_async

        async for chunk in run_agentic_stream_async(
            self,
            user_input=user_input,
            system_prompt=system_prompt,
            tools=tools,
            tool_functions=tool_functions,
            max_iterations=max_iterations,
            model=model,
            tool_choice=tool_choice,
            extra_messages=extra_messages,
            content_separator=content_separator,
            **kwargs,
        ):
            yield chunk

    # ─── Canonical agent methods (AgentLoop / AsyncAgentLoop) ──────────────

    def run_agent(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        budget: Optional["AgentBudget"] = None,
        hooks: Optional["AgentHooks"] = None,
        max_iterations: int = 10,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        deduplicate: bool = False,
        **kwargs: Any,
    ) -> "ModelChatResponse":
        """Execute a ReAct-style agent loop synchronously using AgentLoop.

        This is the canonical sync entry point for agentic workflows. It uses
        the newer ``AgentLoop`` architecture which provides:

        - Budget enforcement (iterations, tokens, wall-clock)
        - Lifecycle hooks (on_iteration_start, on_llm_response, etc.)
        - Parallel tool execution via ThreadPoolExecutor
        - Structured state inspection via ``.state`` property
        - Concurrency guards (RuntimeError on concurrent calls)
        - Tool deduplication (opt-in)

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            tools: Tool definitions (callables, dict specs, or Pydantic models).
            tool_functions: Dict mapping custom names to callables.
            budget: Execution bounds (max_iterations, token limits, timeouts).
                If not provided, a default budget is created from max_iterations.
            hooks: Optional lifecycle callbacks (AgentHooks protocol).
            max_iterations: Hard cap on loop iterations (default: 10).
                Only used when budget is not provided.
            model: Optional model override, passed through to the engine.
            tool_choice: Tool choice directive (default: "auto").
            extra_messages: Optional pre-existing messages before user_input.
            content_separator: String to join content between iterations.
            deduplicate: Enable fingerprint-based tool deduplication.
            **kwargs: Extra args passed through to the engine's generate().

        Returns:
            The final ModelChatResponse with concatenated content.

        Raises:
            RuntimeError: If called while the loop is already running.
            AgentBudgetExceeded: If any budget constraint is violated.

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> def get_weather(city: str) -> dict:
            ...     return {"temp": 72, "unit": "F"}
            >>> response = client.run_agent(
            ...     user_input="What's the weather in London?",
            ...     tools=[get_weather],
            ...     max_iterations=5,
            ... )
            >>> print(response.content)
        """
        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.types import AgentBudget

        if budget is None:
            budget = AgentBudget(max_iterations=max_iterations)

        loop_kwargs = {}
        if model is not None:
            loop_kwargs["model"] = model

        loop = AgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            **loop_kwargs,
        )

        return loop.run(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

    def run_agent_stream(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        budget: Optional["AgentBudget"] = None,
        hooks: Optional["AgentHooks"] = None,
        max_iterations: int = 10,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        deduplicate: bool = False,
        **kwargs: Any,
    ) -> Iterator["ChatCompletionModel"]:
        """Stream chunks from a ReAct-style agent loop synchronously.

        Yields ChatCompletionModel chunks as they arrive from the LLM.
        Between iterations with tool calls, a separator chunk is yielded.

        Uses the newer ``AgentLoop`` architecture with parallel tool execution,
        budget enforcement, and lifecycle hooks.

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            tools: Tool definitions (callables, dict specs, or Pydantic models).
            tool_functions: Dict mapping custom names to callables.
            budget: Execution bounds. Defaults to AgentBudget(max_iterations).
            hooks: Optional lifecycle callbacks.
            max_iterations: Hard cap on loop iterations (default: 10).
            model: Optional model override.
            tool_choice: Tool choice directive (default: "auto").
            extra_messages: Optional pre-existing messages.
            content_separator: String to join content between iterations.
            deduplicate: Enable tool deduplication.
            **kwargs: Extra args passed to stream_generate.

        Yields:
            ChatCompletionModel chunks from the streaming response.

        Raises:
            RuntimeError: If called while the loop is already running.

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> for chunk in client.run_agent_stream("Hello", tools=[my_tool]):
            ...     if chunk.choices[0].delta.content:
            ...         print(chunk.choices[0].delta.content, end="")
        """
        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.types import AgentBudget

        if budget is None:
            budget = AgentBudget(max_iterations=max_iterations)

        loop_kwargs = {}
        if model is not None:
            loop_kwargs["model"] = model

        loop = AgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            **loop_kwargs,
        )

        yield from loop.stream(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

    async def run_agent_async(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        budget: Optional["AgentBudget"] = None,
        hooks: Optional["AgentHooks"] = None,
        max_iterations: int = 10,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        deduplicate: bool = False,
        **kwargs: Any,
    ) -> "ModelChatResponse":
        """Execute a ReAct-style agent loop asynchronously using AsyncAgentLoop.

        This is the canonical async entry point for agentic workflows. It uses
        the newer ``AsyncAgentLoop`` architecture which provides:

        - Async tool execution (supports both sync and async callables)
        - Parallel tool execution via asyncio.gather
        - Budget enforcement (iterations, tokens, wall-clock)
        - Lifecycle hooks (on_iteration_start, on_llm_response, etc.)
        - Structured state inspection via ``.state`` property
        - Concurrency guards (RuntimeError on concurrent calls)
        - Tool deduplication (opt-in)

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            tools: Tool definitions (callables, dict specs, or Pydantic models).
            tool_functions: Dict mapping custom names to callables.
            budget: Execution bounds. Defaults to AgentBudget(max_iterations).
            hooks: Optional lifecycle callbacks.
            max_iterations: Hard cap on loop iterations (default: 10).
            model: Optional model override.
            tool_choice: Tool choice directive (default: "auto").
            extra_messages: Optional pre-existing messages.
            content_separator: String to join content between iterations.
            deduplicate: Enable tool deduplication.
            **kwargs: Extra args passed to async_generate.

        Returns:
            The final ModelChatResponse with concatenated content.

        Raises:
            RuntimeError: If called while the loop is already running.
            AgentBudgetExceeded: If any budget constraint is violated.

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> async def fetch_url(url: str) -> str:
            ...     async with aiohttp.ClientSession() as session:
            ...         async with session.get(url) as resp:
            ...             return await resp.text()
            >>> response = await client.run_agent_async(
            ...     user_input="Fetch example.com",
            ...     tools=[fetch_url],
            ... )
        """
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        from magic_llm.agent.types import AgentBudget

        if budget is None:
            budget = AgentBudget(max_iterations=max_iterations)

        loop_kwargs = {}
        if model is not None:
            loop_kwargs["model"] = model

        loop = AsyncAgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            **loop_kwargs,
        )

        return await loop.run(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

    async def run_agent_stream_async(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List["ToolSpec"]] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        budget: Optional["AgentBudget"] = None,
        hooks: Optional["AgentHooks"] = None,
        max_iterations: int = 10,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        deduplicate: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator["ChatCompletionModel"]:
        """Stream chunks from a ReAct-style agent loop asynchronously.

        Returns an AsyncIterator[ChatCompletionModel]. Yields chunks as they
        arrive from the LLM, with separator chunks between iterations.

        Uses the newer ``AsyncAgentLoop`` architecture with async parallel
        tool execution, budget enforcement, and lifecycle hooks.

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            tools: Tool definitions (callables, dict specs, or Pydantic models).
            tool_functions: Dict mapping custom names to callables.
            budget: Execution bounds. Defaults to AgentBudget(max_iterations).
            hooks: Optional lifecycle callbacks.
            max_iterations: Hard cap on loop iterations (default: 10).
            model: Optional model override.
            tool_choice: Tool choice directive (default: "auto").
            extra_messages: Optional pre-existing messages.
            content_separator: String to join content between iterations.
            deduplicate: Enable tool deduplication.
            **kwargs: Extra args passed to async_stream_generate.

        Yields:
            ChatCompletionModel chunks from the streaming response.

        Raises:
            RuntimeError: If called while the loop is already running.

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> async for chunk in client.run_agent_stream_async("Hello"):
            ...     if chunk.choices[0].delta.content:
            ...         print(chunk.choices[0].delta.content, end="")
        """
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        from magic_llm.agent.types import AgentBudget

        if budget is None:
            budget = AgentBudget(max_iterations=max_iterations)

        loop_kwargs = {}
        if model is not None:
            loop_kwargs["model"] = model

        loop = AsyncAgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            **loop_kwargs,
        )

        async for chunk in loop.stream(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        ):
            yield chunk
