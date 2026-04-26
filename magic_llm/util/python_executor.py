import json
import logging
import subprocess
import sys
import textwrap

logger = logging.getLogger(__name__)

# Builtins to block in restricted mode
_RESTRICTED_BUILTINS_BLOCK = frozenset({
    'open', 'exec', 'eval', 'compile', '__import__', 'input', 'breakpoint',
})


class PythonExecutor:
    """Callable tool that executes Python code with configurable safety modes.

    Registered as a tool via its __call__ method.
    magic-llm's normalize_openai_tools() auto-extracts schema from
    the __call__ signature: (code: str) -> str

    Safety modes:
        - "subprocess" (default): Code runs in a separate process with no access
          to parent memory/state. Configurable timeout enforced.
        - "restricted_builtins": Code runs in-process with a restricted __builtins__
          dict. NOT a security boundary — determined users can escape it.
        - "in_process": Code runs in-process with full builtins. DANGER — requires
          explicit opt-in and logs a WARNING.
    """

    __name__ = "execute_python"

    def __init__(
        self,
        safety_mode: str = "subprocess",
        timeout: float = 30.0,
        max_output_chars: int = 8000,
    ):
        self.safety_mode = safety_mode
        self.timeout = timeout
        self.max_output_chars = max_output_chars

        if self.safety_mode == "in_process":
            logger.warning(
                "PythonExecutor running with safety_mode='in_process' — "
                "arbitrary code execution is enabled. Do not use with untrusted code."
            )

    @property
    def tool_schema(self) -> dict:
        """OpenAI-compatible tool definition dict."""
        return {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code and return stdout output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    @property
    def tool_callable(self):
        """Return self as the callable executor."""
        return self

    def __call__(self, code: str) -> str:
        """Execute Python code and return stdout output.

        Args:
            code: Python source code to execute.

        Returns:
            stdout output as a string, or an error JSON string on failure.
        """
        if self.safety_mode == "subprocess":
            return self._exec_subprocess(code)
        elif self.safety_mode == "restricted_builtins":
            return self._exec_restricted(code)
        else:
            return self._exec_in_process(code)

    def _exec_subprocess(self, code: str) -> str:
        """Execute code in a separate subprocess with timeout."""
        # Use -u for unbuffered output, -c to run code string
        cmd = [sys.executable, "-u", "-c", code]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout
            if result.returncode != 0:
                error_info = result.stderr.strip() if result.stderr else "Unknown error"
                return json.dumps({"error": error_info})
            return self._truncate(output)
        except subprocess.TimeoutExpired:
            return json.dumps({
                "error": f"Execution timed out after {self.timeout}s"
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _exec_restricted(self, code: str) -> str:
        """Execute code in-process with restricted builtins.

        NOTE: This is NOT a security boundary. Determined users can escape
        via object introspection (e.g., ().__class__.__bases__[0].__subclasses__()).
        """
        # Build a restricted builtins dict
        safe_builtins = {}
        for name, value in __builtins__.items() if isinstance(__builtins__, dict) else dir(__builtins__):
            if isinstance(__builtins__, dict):
                if name not in _RESTRICTED_BUILTINS_BLOCK:
                    safe_builtins[name] = value
            else:
                if name not in _RESTRICTED_BUILTINS_BLOCK:
                    safe_builtins[name] = getattr(__builtins__, name)

        # Remove blocked names explicitly
        for blocked in _RESTRICTED_BUILTINS_BLOCK:
            safe_builtins.pop(blocked, None)

        globals_dict = {"__builtins__": safe_builtins}

        try:
            # Capture stdout
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, globals_dict)  # noqa: S102
            return self._truncate(buf.getvalue())
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _exec_in_process(self, code: str) -> str:
        """Execute code in-process with full builtins access.

        WARNING: This allows arbitrary code execution. Only use with trusted code.
        """
        try:
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, {})  # noqa: S102
            return self._truncate(buf.getvalue())
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _truncate(self, output: str) -> str:
        """Truncate output if it exceeds max_output_chars."""
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + (
                f"\n... [truncated {len(output) - self.max_output_chars} chars]"
            )
        return output
