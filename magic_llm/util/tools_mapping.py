import copy
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

try:
    from pydantic import BaseModel  # pydantic v2
except Exception:  # pragma: no cover - optional import safety
    BaseModel = None  # type: ignore

# Types
OpenAITool = Dict[str, Any]
ToolChoice = Union[str, Dict[str, Any], None]


def _is_pydantic_model(obj: Any) -> bool:
    """Return True if obj is a Pydantic BaseModel class or instance."""
    if BaseModel is None:
        return False
    try:
        # class or instance
        return (inspect.isclass(obj) and issubclass(obj, BaseModel)) or isinstance(obj, BaseModel)
    except Exception:
        return False


def _inline_local_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline local $ref entries that point to root $defs/definitions.

    - Resolves references like {"$ref": "#/$defs/Foo"} by replacing with the
      concrete schema from root["$defs"]["Foo"], recursively.
    - Removes top-level $defs/definitions after inlining so the result is a
      standalone schema acceptable for OpenAI tool parameters.
    """
    root = copy.deepcopy(schema)

    # Gather local definitions (Pydantic v2 uses $defs; older may use definitions)
    local_defs = {}
    if isinstance(root, dict):
        if isinstance(root.get("$defs"), dict):
            local_defs.update(root["$defs"])
        if isinstance(root.get("definitions"), dict):
            # prefer $defs keys; but include definitions if present
            for k, v in root["definitions"].items():
                local_defs.setdefault(k, v)

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            # Replace $ref to local defs
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                parts = ref.lstrip("#/").split("/")
                if parts and parts[0] in ("$defs", "definitions") and len(parts) >= 2:
                    key = parts[-1]
                    target = local_defs.get(key)
                    if isinstance(target, dict):
                        resolved = resolve(copy.deepcopy(target))
                        # Merge any sibling keys besides $ref (rare but allowed)
                        for k, v in node.items():
                            if k == "$ref":
                                continue
                            # do not overwrite resolved keys
                            resolved.setdefault(k, v)
                        return resolved
                    else:
                        # Unknown ref; drop $ref and continue processing other keys
                        node = {k: v for k, v in node.items() if k != "$ref"}

            # Recurse into dict members
            for k, v in list(node.items()):
                node[k] = resolve(v)
            return node
        if isinstance(node, list):
            return [resolve(i) for i in node]
        return node

    root = resolve(root)

    # Strip local definitions on the root if present after inlining
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

    # Optional[T] / Union[T, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _json_schema_for_annotation(non_none[0])
        # Generic union -> enum of types (fallback to string)
        return {}

    # List[T] / Sequence[T]
    if origin in (list, List, tuple, set):
        items = _json_schema_for_annotation(args[0]) if args else {}
        return {"type": "array", "items": items or {}}

    # Dict[K, V]
    if origin in (dict, Dict):
        return {"type": "object"}

    # Literal[...] -> enum
    try:
        from typing import Literal  # py39+
    except Exception:  # pragma: no cover
        Literal = None
    if Literal is not None and origin is Literal:
        return {"enum": list(args)}

    # Pydantic model as parameter type
    if _is_pydantic_model(annotation):
        model = annotation if inspect.isclass(annotation) else annotation.__class__
        try:
            schema = model.model_json_schema()  # type: ignore[attr-defined]
        except Exception:
            schema = {}
        # Inline local refs and strip $defs to satisfy OpenAI tool schema expectations
        schema = _inline_local_refs(schema) if isinstance(schema, dict) else {}
        return schema or {"type": "object"}

    # Primitive types
    mapping = {bool: "boolean", int: "integer", float: "number", str: "string"}
    if annotation in mapping:
        return {"type": mapping[annotation]}

    # Fallback
    return {}


def _extract_param_docs_from_docstring(doc: str) -> Dict[str, str]:
    """Extract per-parameter descriptions from a docstring.

    Supports common styles:
    - reST/Sphinx: ":param name: description" (optionally with separate ":type name:")
    - Google/NumPy-like sections: "Args:" / "Arguments:" / "Parameters:" followed by
      "name (type): description" entries (with indented continuations).

    Returns a mapping of parameter name -> description. Missing entries are ignored.
    """
    if not doc:
        return {}

    lines = doc.expandtabs().splitlines()
    docs: Dict[str, str] = {}

    # Pass 1: reST/Sphinx style ":param ...: ..."
    rst_param_re = re.compile(r"^\s*:param\s+(?:[^:]+?\s+)?(?P<name>\*{0,2}[\w]+)\s*:\s*(?P<desc>.*)\s*$")
    i = 0
    while i < len(lines):
        m = rst_param_re.match(lines[i])
        if m:
            name = m.group("name").strip()
            desc = m.group("desc").strip()
            # Continuation lines: keep consuming indented lines until another ":field:" or header
            j = i + 1
            cont: List[str] = []
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\s*:\w", nxt):  # next field like :type, :return:, etc.
                    break
                if re.match(r"^\s*(Returns?|Yields?|Raises?)\s*:\s*$", nxt):
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

    # Pass 2: Google/"Args:" or NumPy-like "Parameters:" section
    section_header_re = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
    i = 0
    while i < len(lines):
        if not section_header_re.match(lines[i]):
            i += 1
            continue
        base_indent = len(re.match(r"^(\s*)", lines[i]).group(1))
        j = i + 1
        # Allow optional underline (NumPy style): a line of dashes
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
            # name (type): desc  OR  name: desc
            m = re.match(r"^(?P<name>\*{0,2}[\w]+)\s*(\([^)]*\))?\s*:\s*(?P<desc>.*)$", stripped)
            if m:
                name = m.group("name").strip()
                desc = m.group("desc").strip()
                k = j + 1
                # Continuations are more indented than this line
                cont: List[str] = []
                while k < len(lines):
                    nxt = lines[k]
                    ni = len(re.match(r"^(\s*)", nxt).group(1))
                    if ni <= this_indent:
                        break
                    cont.append(nxt.strip())
                    k += 1
                if cont:
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
    # Extract per-parameter descriptions from the docstring, if any
    param_docs = _extract_param_docs_from_docstring(description)

    sig = inspect.signature(fn)
    type_hints = {}
    try:
        from typing import get_type_hints as _get_type_hints

        type_hints = _get_type_hints(fn)
    except Exception:
        pass

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        ann = type_hints.get(pname, param.annotation)
        schema = _json_schema_for_annotation(ann)
        properties[pname] = schema or {"type": "string"}
        # Attach description from docstring when available
        pdesc = param_docs.get(pname)
        if pdesc:
            # Do not overwrite if a description is already present
            if not isinstance(properties[pname], dict):
                properties[pname] = {"description": pdesc}
            else:
                properties[pname].setdefault("description", pdesc)

        is_optional = False
        origin = get_origin(ann)
        args = get_args(ann)
        if origin is Union and any(a is type(None) for a in args):  # noqa: E721
            is_optional = True

        if param.default is inspect._empty and not is_optional:
            required.append(pname)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return name, description, parameters


def _schema_from_pydantic(model_obj: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Extract (name, description, parameters_schema) from a Pydantic model class or instance."""
    model_cls = model_obj if inspect.isclass(model_obj) else model_obj.__class__
    name = getattr(model_cls, "__name__", "tool")
    description = (inspect.getdoc(model_cls) or "").strip()
    try:
        schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
    except Exception:
        schema = {"type": "object", "properties": {}, "required": []}
    # Ensure object schema
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {}, "required": []}
    else:
        # Inline local refs and strip $defs to satisfy OpenAI tool schema expectations
        schema = _inline_local_refs(schema)
    return name, description, schema


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

        # --- Pydantic model class or instance -------------------------------
        if _is_pydantic_model(tool):
            name, desc, params = _schema_from_pydantic(tool)
            normalized.append({"name": name, "description": desc, "parameters": params})
            continue

        # --- Python callable -------------------------------------------------
        if callable(tool) and not isinstance(tool, dict):
            try:
                name, desc, params = _schema_from_callable(tool)
            except Exception:
                continue
            normalized.append({"name": name, "description": desc, "parameters": params})
            continue

        # --- Dict inputs -----------------------------------------------------
        if isinstance(tool, dict):
            # If wrapped in {type:function, function:{...}}
            if tool.get("type") == "function":
                fn_def = copy.deepcopy(tool.get("function", {}))
                # propagate any flags like strict
                if "strict" in tool and "strict" not in fn_def:
                    fn_def["strict"] = tool["strict"]
            else:
                # Legacy or mixed: tool itself is the function definition
                fn_def = copy.deepcopy(tool)

            # Support synonyms for parameters
            if "parameters" not in fn_def:
                if "schema" in fn_def:
                    fn_def["parameters"] = fn_def["schema"]
                elif "input_schema" in fn_def:
                    fn_def["parameters"] = fn_def["input_schema"]

            # If only a raw JSON Schema was provided, try to derive the name
            name = fn_def.get("name") or fn_def.get("title")
            if not name:
                # Cannot normalise without a name
                continue

            params = fn_def.get("parameters") or {
                "type": "object",
                "properties": {},
                "required": [],
            }
            # Ensure parameters is an object schema
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {
                    "type": "object",
                    "properties": params if isinstance(params, dict) else {},
                    "required": [],
                }

            entry = {
                "name": name,
                "description": fn_def.get("description", ""),
                "parameters": params,
            }
            # Preserve optional strict flag when present
            if "strict" in fn_def:
                entry["strict"] = fn_def["strict"]

            normalized.append(entry)
            continue

        # Unknown type – skip
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
        openai_tools = []
        for t in normalized_tools:
            func_block = {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            # If strict was preserved in normalization, include it inside the function block
            if t.get("strict") is not None:
                func_block["strict"] = t["strict"]

            tool_item = {
                "type": "function",
                "function": func_block,
            }
            openai_tools.append(tool_item)

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
