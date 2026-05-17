"""Engine/core-owned tooling boundary.

This module is the canonical owner for tool normalization, provider request
mapping, response/stream tool-call normalization, provider-correct tool-result
message construction, and package-path tooling guardrails.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin

from magic_llm.exception.ChatException import ChatException

try:
    from magic_llm.agent.types import CanonicalToolCall, ToolResult
except Exception:  # pragma: no cover - import safety for tooling-only usage
    CanonicalToolCall = Any  # type: ignore
    ToolResult = Any  # type: ignore

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional import safety
    BaseModel = None  # type: ignore


OpenAITool = Dict[str, Any]
ToolChoice = Union[str, Dict[str, Any], None]
ProviderKey = Literal["openai", "anthropic", "google", "deepinfra"]


@dataclass
class RequestTools:
    tools: Any | None = None
    tool_choice: Any | None = None


@dataclass
class StreamIterationSummary:
    content: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str | None = None
    last_chunk: Any | None = None


@dataclass
class AnthropicStreamState:
    active_blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    idx: str | None = None
    usage: Any | None = None


def _is_pydantic_model(obj: Any) -> bool:
    """Return True if obj is a Pydantic BaseModel class or instance."""
    if BaseModel is None:
        return False
    try:
        return (inspect.isclass(obj) and issubclass(obj, BaseModel)) or isinstance(obj, BaseModel)
    except Exception:
        return False


def _inline_local_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline local $ref entries that point to root $defs/definitions."""
    root = copy.deepcopy(schema)
    local_defs = {}
    if isinstance(root, dict):
        if isinstance(root.get("$defs"), dict):
            local_defs.update(root["$defs"])
        if isinstance(root.get("definitions"), dict):
            for k, v in root["definitions"].items():
                local_defs.setdefault(k, v)

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                parts = ref.lstrip("#/").split("/")
                if parts and parts[0] in ("$defs", "definitions") and len(parts) >= 2:
                    target = local_defs.get(parts[-1])
                    if isinstance(target, dict):
                        resolved = resolve(copy.deepcopy(target))
                        for k, v in node.items():
                            if k != "$ref":
                                resolved.setdefault(k, v)
                        return resolved
                    node = {k: v for k, v in node.items() if k != "$ref"}
            for k, v in list(node.items()):
                node[k] = resolve(v)
            return node
        if isinstance(node, list):
            return [resolve(i) for i in node]
        return node

    root = resolve(root)
    if isinstance(root, dict):
        root.pop("$defs", None)
        root.pop("definitions", None)
    return root


def _json_schema_for_annotation(annotation: Any) -> Dict[str, Any]:
    """Best-effort conversion from a Python type annotation to a JSON Schema snippet."""
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _json_schema_for_annotation(non_none[0])
        return {}

    if origin in (list, List, tuple, set):
        items = _json_schema_for_annotation(args[0]) if args else {}
        return {"type": "array", "items": items or {}}

    if origin in (dict, Dict):
        return {"type": "object"}

    try:
        from typing import Literal
    except Exception:  # pragma: no cover
        Literal = None
    if Literal is not None and origin is Literal:
        return {"enum": list(args)}

    if _is_pydantic_model(annotation):
        model = annotation if inspect.isclass(annotation) else annotation.__class__
        try:
            schema = model.model_json_schema()  # type: ignore[attr-defined]
        except Exception:
            schema = {}
        schema = _inline_local_refs(schema) if isinstance(schema, dict) else {}
        return schema or {"type": "object"}

    mapping = {bool: "boolean", int: "integer", float: "number", str: "string"}
    if annotation in mapping:
        return {"type": mapping[annotation]}
    return {}


def _extract_param_docs_from_docstring(doc: str) -> Dict[str, str]:
    """Extract per-parameter descriptions from common docstring styles."""
    if not doc:
        return {}
    lines = doc.expandtabs().splitlines()
    docs: Dict[str, str] = {}

    rst_param_re = re.compile(r"^\s*:param\s+(?:[^:]+?\s+)?(?P<name>\*{0,2}[\w]+)\s*:\s*(?P<desc>.*)\s*$")
    i = 0
    while i < len(lines):
        m = rst_param_re.match(lines[i])
        if m:
            name = m.group("name").strip()
            desc = m.group("desc").strip()
            j = i + 1
            cont: List[str] = []
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\s*:\w", nxt) or re.match(r"^\s*(Returns?|Yields?|Raises?)\s*:\s*$", nxt):
                    break
                if nxt.strip() == "":
                    cont.append("")
                    j += 1
                    continue
                if nxt.startswith(" "):
                    cont.append(nxt.strip())
                    j += 1
                    continue
                break
            extra = " ".join([c for c in cont if c.strip() != ""]) if cont else ""
            if extra:
                desc = (desc + " " + extra).strip()
            if name and desc:
                docs[name] = desc
            i = j
            continue
        i += 1

    section_header_re = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
    i = 0
    while i < len(lines):
        if not section_header_re.match(lines[i]):
            i += 1
            continue
        base_indent = len(re.match(r"^(\s*)", lines[i]).group(1))
        j = i + 1
        if j < len(lines) and re.match(r"^\s*-{3,}\s*$", lines[j]):
            j += 1
        while j < len(lines):
            line = lines[j]
            if re.match(r"^\s*(Returns?|Yields?|Raises?)\s*:\s*$", line):
                break
            this_indent = len(re.match(r"^(\s*)", line).group(1))
            if this_indent <= base_indent:
                break
            stripped = line.strip()
            if not stripped:
                j += 1
                continue
            m = re.match(r"^(?P<name>\*{0,2}[\w]+)\s*(\([^)]*\))?\s*:\s*(?P<desc>.*)$", stripped)
            if m:
                name = m.group("name").strip()
                desc = m.group("desc").strip()
                k = j + 1
                cont: List[str] = []
                while k < len(lines):
                    nxt = lines[k]
                    ni = len(re.match(r"^(\s*)", nxt).group(1))
                    if ni <= this_indent:
                        break
                    cont.append(nxt.strip())
                    k += 1
                extra = " ".join([c for c in cont if c.strip() != ""]) if cont else ""
                if extra:
                    desc = (desc + " " + extra).strip()
                if name and desc and name not in docs:
                    docs[name] = desc
                j = k
                continue
            j += 1
        i = j
    return docs


def _schema_from_callable(fn: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Extract (name, description, parameters_schema) from a Python callable."""
    name = getattr(fn, "__name__", None) or "tool"
    description = (inspect.getdoc(fn) or "").strip()
    param_docs = _extract_param_docs_from_docstring(description)
    sig = inspect.signature(fn)
    try:
        from typing import get_type_hints as _get_type_hints
        type_hints = _get_type_hints(fn)
    except Exception:
        type_hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        ann = type_hints.get(pname, param.annotation)
        schema = _json_schema_for_annotation(ann)
        properties[pname] = schema or {"type": "string"}
        if pdesc := param_docs.get(pname):
            if not isinstance(properties[pname], dict):
                properties[pname] = {"description": pdesc}
            else:
                properties[pname].setdefault("description", pdesc)
        origin = get_origin(ann)
        args = get_args(ann)
        is_optional = origin is Union and any(a is type(None) for a in args)  # noqa: E721
        if param.default is inspect._empty and not is_optional:
            required.append(pname)

    return name, description, {"type": "object", "properties": properties, "required": required}


def _schema_from_pydantic(model_obj: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Extract (name, description, parameters_schema) from a Pydantic model."""
    model_cls = model_obj if inspect.isclass(model_obj) else model_obj.__class__
    name = getattr(model_cls, "__name__", "tool")
    description = (inspect.getdoc(model_cls) or "").strip()
    try:
        schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
    except Exception:
        schema = {"type": "object", "properties": {}, "required": []}
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {}, "required": []}
    else:
        schema = _inline_local_refs(schema)
    return name, description, schema


def normalize_openai_tools(tools: Optional[List[OpenAITool]]) -> List[Dict[str, Any]]:
    """Normalize supported raw tool definitions to function definitions."""
    if not tools:
        return []
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if tool is None:
            continue
        if _is_pydantic_model(tool):
            name, desc, params = _schema_from_pydantic(tool)
            normalized.append({"name": name, "description": desc, "parameters": params})
            continue
        if callable(tool) and not isinstance(tool, dict):
            try:
                name, desc, params = _schema_from_callable(tool)
            except Exception as exc:
                raise ChatException(
                    message=f"Unsupported tool definition {tool!r}: {exc}",
                    error_code="TOOL_NORMALIZATION_ERROR",
                ) from exc
            normalized.append({"name": name, "description": desc, "parameters": params})
            continue
        if isinstance(tool, dict):
            if tool.get("type") == "function":
                fn_def = copy.deepcopy(tool.get("function", {}))
                if "strict" in tool and "strict" not in fn_def:
                    fn_def["strict"] = tool["strict"]
            else:
                fn_def = copy.deepcopy(tool)
            if "parameters" not in fn_def:
                if "schema" in fn_def:
                    fn_def["parameters"] = fn_def["schema"]
                elif "input_schema" in fn_def:
                    fn_def["parameters"] = fn_def["input_schema"]
            name = fn_def.get("name") or fn_def.get("title")
            if not name:
                raise ChatException(
                    message=f"Unsupported tool definition {tool!r}: missing tool name",
                    error_code="TOOL_NORMALIZATION_ERROR",
                )
            params = fn_def.get("parameters") or {"type": "object", "properties": {}, "required": []}
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object", "properties": params if isinstance(params, dict) else {}, "required": []}
            entry = {"name": name, "description": fn_def.get("description", ""), "parameters": params}
            if "strict" in fn_def:
                entry["strict"] = fn_def["strict"]
            normalized.append(entry)
            continue
        raise ChatException(
            message=f"Unsupported tool definition {tool!r}: unsupported shape {type(tool).__name__}",
            error_code="TOOL_NORMALIZATION_ERROR",
        )
    return normalized


def normalize_openai_tool_choice(tool_choice: ToolChoice) -> ToolChoice:
    """Normalize tool_choice to common string or {'name': name} form."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = tool_choice.get("function", {}).get("name")
            return {"name": name} if name else None
        if "name" in tool_choice:
            return {"name": tool_choice["name"]}
    return None


def map_to_anthropic(tools: Optional[List[OpenAITool]], tool_choice: ToolChoice) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    normalized_tools = normalize_openai_tools(tools)
    normalized_choice = normalize_openai_tool_choice(tool_choice)
    anthropic_tools = None
    if normalized_tools:
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}, "required": []}),
            }
            for t in normalized_tools
        ]
    anthropic_choice = None
    if isinstance(normalized_choice, str):
        if normalized_choice == "auto":
            anthropic_choice = {"type": "auto"}
        elif normalized_choice == "required":
            anthropic_choice = {"type": "any"}
        elif normalized_choice == "none":
            anthropic_choice = None
    elif isinstance(normalized_choice, dict) and normalized_choice.get("name"):
        anthropic_choice = {"type": "tool", "name": normalized_choice["name"]}
    return anthropic_tools, anthropic_choice


def map_to_openai(tools: Optional[List[OpenAITool]], tool_choice: ToolChoice) -> Tuple[Optional[List[Dict[str, Any]]], ToolChoice]:
    normalized_tools = normalize_openai_tools(tools)
    normalized_choice = normalize_openai_tool_choice(tool_choice)
    openai_tools = None
    if normalized_tools:
        openai_tools = []
        for t in normalized_tools:
            func_block = {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}, "required": []}),
            }
            if t.get("strict") is not None:
                func_block["strict"] = t["strict"]
            openai_tools.append({"type": "function", "function": func_block})
    openai_choice: ToolChoice = None
    if isinstance(normalized_choice, str):
        openai_choice = normalized_choice
    elif isinstance(normalized_choice, dict) and normalized_choice.get("name"):
        openai_choice = {"type": "function", "function": {"name": normalized_choice["name"]}}
    return openai_tools, openai_choice


def map_to_gemini(tools: Optional[List[OpenAITool]], tool_choice: ToolChoice) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    normalized_tools = normalize_openai_tools(tools)
    normalized_choice = normalize_openai_tool_choice(tool_choice)
    gemini_tools = None
    if normalized_tools:
        gemini_tools = []
        for t in normalized_tools:
            params = t.get("parameters", {"type": "object", "properties": {}, "required": []})
            if isinstance(params, dict):
                params = _inline_local_refs(params)
                params["additionalProperties"] = False
            gemini_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parametersJsonSchema": params,
            })
    gemini_tool_config = None
    if isinstance(normalized_choice, str):
        mode_map = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
        if mode := mode_map.get(normalized_choice):
            gemini_tool_config = {"functionCallingConfig": {"mode": mode}}
    elif isinstance(normalized_choice, dict) and normalized_choice.get("name"):
        gemini_tool_config = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [normalized_choice["name"]]}}
    return gemini_tools, gemini_tool_config


def coerce_tool_choice_to_string(tool_choice: ToolChoice, default: str = "auto") -> Optional[str]:
    """Compatibility helper for legacy callers. Named choices are not preserved."""
    normalized = normalize_openai_tool_choice(tool_choice)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return normalized
    return default


def map_request_tools(provider: str, tools: list[Any] | None, tool_choice: ToolChoice) -> RequestTools:
    """Map raw package tool definitions to provider request shape exactly once."""
    provider_key = provider.lower()
    if provider_key in {"openai", "deepinfra"}:
        mapped_tools, mapped_choice = map_to_openai(tools, tool_choice)
        return RequestTools(tools=mapped_tools, tool_choice=mapped_choice)
    if provider_key == "anthropic":
        mapped_tools, mapped_choice = map_to_anthropic(tools, tool_choice)
        return RequestTools(tools=mapped_tools, tool_choice=mapped_choice)
    if provider_key in {"google", "gemini"}:
        mapped_tools, mapped_choice = map_to_gemini(tools, tool_choice)
        return RequestTools(
            tools=[{"functionDeclarations": mapped_tools}] if mapped_tools else None,
            tool_choice=mapped_choice,
        )
    mapped_tools, mapped_choice = map_to_openai(tools, tool_choice)
    return RequestTools(tools=mapped_tools, tool_choice=mapped_choice)


def extract_tool_calls(response: Any) -> list[Any]:
    """Parse normalized ModelChatResponse.tool_calls into CanonicalToolCall-compatible calls."""
    raw_calls = getattr(response, "tool_calls", None)
    if not raw_calls:
        return []
    result = []
    for tc in raw_calls:
        function = getattr(tc, "function", None)
        if function is None and isinstance(tc, dict):
            function = tc.get("function")
        if function is None:
            continue
        arguments_raw = getattr(function, "arguments", None)
        name = getattr(function, "name", None)
        if isinstance(function, dict):
            arguments_raw = function.get("arguments")
            name = function.get("name")
        try:
            arguments = json.loads(arguments_raw or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
        result.append(CanonicalToolCall(id=tc_id or "", name=name or "", arguments=arguments))
    return result


def is_finished(provider: str, response: Any) -> bool:
    """Provider-aware completion check over normalized response data."""
    if provider.lower() in {"google", "gemini"}:
        return not getattr(response, "tool_calls", None)
    return getattr(response, "finish_reason", None) == "stop"


def append_tool_results(provider: str, chat: Any, results: list[Any]) -> None:
    """Append provider-correct tool result messages to ModelChat."""
    provider_key = provider.lower()
    if provider_key == "anthropic":
        _validate_result_completeness(results, chat, "tool_use_id")
        chat.add_tool_messages([{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": result.tool_call_id or "", "content": result.content}
                for result in results
            ],
        }])
        return
    if provider_key in {"google", "gemini"}:
        _validate_result_completeness(results, chat, "tool_call_id")
        chat.add_tool_messages([{
            "role": "user",
            "content": [
                {
                    "functionResponse": {
                        "id": result.tool_call_id or "",
                        "name": result.name,
                        "response": {"error": result.content} if result.is_error else {"output": result.content},
                    }
                }
                for result in results
            ],
        }])
        return
    for result in results:
        chat.add_tool_result(
            tool_call_id=result.tool_call_id or "",
            content=result.content,
            is_error=result.is_error,
        )


def _last_assistant_tool_call_ids(chat: Any) -> set[str]:
    for msg in reversed(chat.messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = set()
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    ids.add(tc_id)
            return ids
    return set()


def _validate_result_completeness(results: list[Any], chat: Any, label: str) -> None:
    expected = _last_assistant_tool_call_ids(chat)
    if not expected:
        return
    actual = {result.tool_call_id for result in results if getattr(result, "tool_call_id", None)}
    missing = expected - actual
    if missing:
        raise ValueError(f"Incomplete tool results: missing {label}(s): {', '.join(sorted(missing))}")


def validate_tool_result_integrity(provider: str, chat: Any, results: list[Any] | None = None) -> bool:
    """Validate provider pair/completeness rules before next LLM call."""
    expected = _last_assistant_tool_call_ids(chat)
    if not expected:
        return True
    provider_key = provider.lower()
    actual: set[str] = set()
    if provider_key in {"anthropic", "google", "gemini"}:
        for msg in chat.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if not isinstance(part, dict):
                        continue
                    if "functionResponse" in part and part["functionResponse"].get("id"):
                        actual.add(part["functionResponse"]["id"])
                    if part.get("type") == "tool_result" and part.get("tool_use_id"):
                        actual.add(part["tool_use_id"])
    else:
        for msg in chat.messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                actual.add(msg["tool_call_id"])
    if results:
        actual.update(result.tool_call_id for result in results if getattr(result, "tool_call_id", None))
    return expected.issubset(actual)


def _is_actionable_tool_choice(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice not in {"", "none"}
    if isinstance(tool_choice, dict):
        return bool(tool_choice)
    return bool(tool_choice)


def guard_tools_supported(engine_name: str, tools: Any = None, tool_choice: Any = None) -> None:
    """Raise when package path has no current tool request/response wiring."""
    has_tools = bool(tools)
    has_choice = _is_actionable_tool_choice(tool_choice)
    if not has_tools and not has_choice:
        return
    raise ChatException(
        message=(
            f"Tool wiring is not implemented for the {engine_name} package path in magic-llm. "
            "This is a package-side wiring limitation, not a provider capability statement."
        ),
        error_code="TOOL_WIRING_NOT_IMPLEMENTED",
    )


def infer_provider_from_client(client: Any) -> str:
    """Infer engine/provider key from a MagicLLM-like client."""
    engine_instance = getattr(client, "llm", None)
    if engine_instance is None:
        return getattr(client, "engine", "openai")
    return getattr(type(engine_instance), "engine", "openai")


def accumulate_stream_chunk(summary: StreamIterationSummary, chunk: Any) -> StreamIterationSummary:
    """Accumulate normalized stream chunks into a provider-agnostic summary."""
    summary.last_chunk = chunk
    if not getattr(chunk, "choices", None):
        return summary
    choice = chunk.choices[0]
    delta = getattr(choice, "delta", None)
    if getattr(choice, "finish_reason", None):
        summary.finish_reason = choice.finish_reason
    if delta and getattr(delta, "content", None):
        summary.content += delta.content
    if delta and getattr(delta, "tool_calls", None):
        _merge_stream_tool_calls(summary.tool_calls, delta.tool_calls)
    return summary


def _merge_stream_tool_calls(accumulated: list[dict[str, Any]], tool_calls: list[Any]) -> None:
    by_index = {entry.get("index", i): entry for i, entry in enumerate(accumulated)}
    for pos, tc in enumerate(tool_calls):
        idx = getattr(tc, "index", None)
        idx = pos if idx is None else idx
        entry = by_index.get(idx)
        if entry is None:
            entry = {"index": idx, "function": {"arguments": ""}}
            by_index[idx] = entry
            accumulated.append(entry)
        if getattr(tc, "id", None):
            entry["id"] = tc.id
        function = getattr(tc, "function", None)
        if function:
            entry.setdefault("function", {})
            if getattr(function, "name", None):
                entry["function"]["name"] = function.name
            if getattr(function, "arguments", None):
                entry["function"]["arguments"] = entry["function"].get("arguments", "") + function.arguments


def stream_summary_tool_calls(summary: StreamIterationSummary) -> list[Any]:
    """Return CanonicalToolCall-compatible calls from an accumulated stream summary."""
    calls = []
    for entry in sorted(summary.tool_calls, key=lambda item: item.get("index", 0)):
        function = entry.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        calls.append(CanonicalToolCall(id=entry.get("id", ""), name=function.get("name", ""), arguments=arguments))
    return calls
