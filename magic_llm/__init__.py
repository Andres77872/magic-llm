import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from magic_llm.base import MagicLlmBase
from magic_llm.model.discovery import NormalizedDiscoveredModel

if TYPE_CHECKING:
    from magic_llm.model.ModelChatResponse import ModelChatResponse
    from magic_llm.model.ModelChatStream import ChatCompletionModel
    from magic_llm.util.agentic import ToolSpec
    from magic_llm.agent.types import AgentBudget, TaskManifest
    from magic_llm.agent.hooks import AgentHooks
    from magic_llm.agent.task_executor import TaskExecutor
    from magic_llm.agent.registry import SubagentRegistry
    from magic_llm.agent.bundle import SubagentBundle
    from magic_llm.agent.definitions import SubagentManifest

__version__ = '0.1.34'  # Subagent architecture complete (definitions/loader/registry/binder/bundle/decorator/config)

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
        # Task executor for task/subagent registration (lazy init)
        self._task_executor: Optional["TaskExecutor"] = None
        # Instance-scoped subagent registry (NO GLOBAL STATE)
        self._subagent_registry: Optional["SubagentRegistry"] = None

    # ─── Task/Subagent Registration API ────────────────────────────────────────

    def register_task(
        self,
        manifest: "TaskManifest",
        callable: Callable[..., Any],
    ) -> None:
        """Register a task/subagent for agent loop execution.

        This is the primary API for wrappers like magic-agents to register
        task subagents with runtime safeguards (depth, timeout, concurrency).

        The registered task becomes available as a tool in subsequent
        run_agent_async() calls when the internal TaskExecutor is used.

        Args:
            manifest: TaskManifest with policy (id, timeout, concurrency, depth).
            callable: Async callable to execute (graph-side implementation).

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> from magic_llm.agent.types import TaskManifest
            >>> manifest = TaskManifest(
            ...     id="web_search",
            ...     name="Web Search",
            ...     description="Search the web",
            ...     input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ...     timeout_seconds=30,
            ...     max_concurrency=5,
            ...     max_depth=3,
            ... )
            >>> async def search(query: str) -> dict:
            ...     return {"results": [...]}
            >>> client.register_task(manifest, search)
            >>> # Now run_agent_async can use 'web_search' as a tool
        """
        if self._task_executor is None:
            from magic_llm.agent.task_executor import TaskExecutor
            # Pass self (MagicLLM instance) for nested LLM node execution
            self._task_executor = TaskExecutor(client=self)
        self._task_executor.register_task(manifest, callable)

    # ─── Unified Subagent Loading API ────────────────────────────────────────

    async def load_subagents(
        self,
        manifest_dir: Path,
        code_registry: Optional[Dict[str, Callable]] = None,
    ) -> "SubagentBundle":
        """Unified entrypoint for subagent loading.

        Orchestrates complete subagent lifecycle:
        1. Discovery: ManifestLoader.load_all(manifest_dir)
        2. Binding: Binder.join() for each manifest + callable
        3. Registration: TaskExecutor.register_task() with safeguards

        magic-llm is repo-agnostic — caller passes explicit manifest_dir.
        NO global state — registry is instance-scoped.

        Args:
            manifest_dir: Directory containing *.agent.yaml files (explicit path).
            code_registry: Dict populated by @subagent decorator (optional).

        Returns:
            SubagentBundle with schemas for agent loop injection.

        Example:
            >>> client = MagicLLM(engine="openai", model="gpt-4")
            >>> code_registry = {}
            >>>
            >>> @subagent("research.web", registry=code_registry)
            >>> async def research_web(query: str) -> str:
            >>>     return execute_agent_loop(...)
            >>>
            >>> bundle = await client.load_subagents(
            >>>     manifest_dir=Path("subagents"),
            >>>     code_registry=code_registry
            >>> )
            >>> # Now run_agent_async can use registered subagents
        """
        from magic_llm.agent.config import is_subagents_enabled
        from magic_llm.agent.registry import SubagentRegistry
        from magic_llm.agent.loader import ManifestLoader
        from magic_llm.agent.binder import Binder
        from magic_llm.agent.bundle import SubagentBundle
        from magic_llm.agent.types import TaskManifest

        # Check feature flag at repo-level
        if not is_subagents_enabled():
            logger.debug("Subagents feature disabled at repo-level — returning empty bundle")
            return SubagentBundle()

        # Initialize instance-scoped registry (NO GLOBAL STATE)
        if self._subagent_registry is None:
            self._subagent_registry = SubagentRegistry()

        registry = self._subagent_registry

        # 1. Discovery: load manifests from explicit directory
        loader = ManifestLoader(manifest_dir)
        manifests = await loader.load_all()

        # 2. Register manifests
        for manifest in manifests:
            registry.register_manifest(manifest)

        # 3. Register callables from code_registry dict
        if code_registry:
            for agent_id, callable in code_registry.items():
                registry.register_callable(agent_id, callable)

        # 4. Bind and register with TaskExecutor
        registered_count = 0
        for manifest in registry.list_manifests():
            callable = registry.get_callable(manifest.id)
            if callable is None:
                logger.warning(f"No callable for manifest '{manifest.id}'")
                continue

            # Bind (signature validation)
            bound_manifest, bound_callable = Binder.join(manifest, callable)

            # Convert to runtime manifest
            task_manifest = self._to_task_manifest(bound_manifest)

            # Register with TaskExecutor (creates semaphore, wraps callable)
            self.register_task(task_manifest, bound_callable)
            registered_count += 1

        registry.mark_initialized()

        # 5. Build bundle for agent loop
        bundle = SubagentBundle.from_registry(registry)

        logger.info(
            f"Loaded {registered_count} subagents from {manifest_dir} "
            f"(bundle has {bundle.registered_count} ready for injection)"
        )

        return bundle

    def _to_task_manifest(
        self,
        subagent_manifest: "SubagentManifest"
    ) -> "TaskManifest":
        """Convert SubagentManifest to TaskManifest for runtime.

        Excludes YAML-specific fields (apiVersion, kind, version, source_file).
        TaskManifest is the runtime subset used by TaskExecutor.

        Args:
            subagent_manifest: SubagentManifest from YAML discovery.

        Returns:
            TaskManifest for TaskExecutor.register_task().
        """
        from magic_llm.agent.types import TaskManifest

        return TaskManifest(
            id=subagent_manifest.id,
            name=subagent_manifest.name,
            description=subagent_manifest.description,
            input_schema=subagent_manifest.input_schema,
            timeout_seconds=subagent_manifest.timeout_seconds,
            max_concurrency=subagent_manifest.max_concurrency,
            max_depth=subagent_manifest.max_depth,
        )

    def reset_depths(self) -> None:
        """Reset depth counters for new graph execution.

        Called internally by TaskExecutor or externally by wrapper.
        Depth tracking uses ContextVar for async task isolation.
        """
        from magic_llm.agent.task_executor import reset_depths
        reset_depths()

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
            NotImplementedError: The tags dictionary download functionality is not yet implemented
        """
        logger.warning("The download_tags_dictionary method is not yet implemented")
        raise NotImplementedError("The tags dictionary download functionality is not yet implemented")

    # ─── Model Discovery ──────────────────────────────────────────────────

    def list_models(self) -> List[NormalizedDiscoveredModel]:
        """Discover available models from the provider's listing API.

        Delegates to the underlying chat engine's ``list_models()``, which
        resolves the registered discovery adapter for this engine and calls
        its ``discover()`` method.

        Returns:
            Normalized list of discovered models. Always returns a list —
            never ``None``.

        Raises:
            NotImplementedError: The engine has no registered discovery
                adapter.
            DiscoveryError: Provider API unreachable or returned an error.
            DiscoveryAuthError: Invalid credentials.
            DiscoveryRateLimitError: Rate limited.

        Example:
            >>> client = MagicLLM(engine="openai", private_key="sk-...")
            >>> models = client.list_models()
            >>> for m in models:
            ...     print(m.external_id, m.capabilities.chat)
        """
        return self.llm.list_models()

    async def async_list_models(self) -> List[NormalizedDiscoveredModel]:
        """Async variant of :meth:`list_models`.

        Calls ``await self.llm.async_list_models()`` which delegates to
        the registered discovery adapter's ``async_discover()``.

        Returns:
            Normalized list of discovered models.

        Raises:
            NotImplementedError: The engine has no registered discovery
                adapter.
            DiscoveryError: Provider API unreachable or returned an error.
            DiscoveryAuthError: Invalid credentials.
            DiscoveryRateLimitError: Rate limited.

        Example:
            >>> client = MagicLLM(engine="openai", private_key="sk-...")
            >>> models = await client.async_list_models()
            >>> print(f"Found {len(models)} models")
        """
        return await self.llm.async_list_models()

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
        max_iterations: int = 150,
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
            max_iterations: Hard cap on loop iterations (default: 150).
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
        max_iterations: int = 150,
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
            max_iterations: Hard cap on loop iterations (default: 150).
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
        max_iterations: int = 150,
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        content_separator: str = "\n\n",
        deduplicate: bool = False,
        task_executor: Optional["TaskExecutor"] = None,
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
        - Task subagent support via TaskExecutor (depth, timeout, concurrency)

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            tools: Tool definitions (callables, dict specs, or Pydantic models).
            tool_functions: Dict mapping custom names to callables.
            budget: Execution bounds. Defaults to AgentBudget(max_iterations).
            hooks: Optional lifecycle callbacks.
            max_iterations: Hard cap on loop iterations (default: 150).
            model: Optional model override.
            tool_choice: Tool choice directive (default: "auto").
            extra_messages: Optional pre-existing messages.
            content_separator: String to join content between iterations.
            deduplicate: Enable tool deduplication.
            task_executor: Optional TaskExecutor override. If None and tasks were
                registered via register_task(), uses the internal TaskExecutor.
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
            ...         ...     return await resp.text()
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

        # Use provided task_executor, or internal one if tasks were registered
        executor = task_executor or self._task_executor

        loop = AsyncAgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            tool_executor=executor,
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
        task_executor: Optional["TaskExecutor"] = None,
        budget: Optional["AgentBudget"] = None,
        hooks: Optional["AgentHooks"] = None,
        max_iterations: int = 150,
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
            task_executor: Optional TaskExecutor override. If None and tasks were
                registered via register_task(), uses the internal TaskExecutor.
            budget: Execution bounds. Defaults to AgentBudget(max_iterations).
            hooks: Optional lifecycle callbacks.
            max_iterations: Hard cap on loop iterations (default: 150).
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

        # Use provided task_executor, or internal one if tasks were registered
        executor = task_executor or self._task_executor

        loop = AsyncAgentLoop(
            client=self,
            tools=tools,
            tool_functions=tool_functions,
            budget=budget,
            hooks=hooks,
            content_separator=content_separator,
            tool_choice=tool_choice,
            deduplicate=deduplicate,
            tool_executor=executor,
            **loop_kwargs,
        )

        async for chunk in loop.stream(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        ):
            yield chunk
