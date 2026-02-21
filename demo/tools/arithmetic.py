"""
Arithmetic tool for verl-tool.

Handles OpenAI-format tool calls:
    <tool_call>{"name": "add", "arguments": {"a": 3, "b": 5}}</tool_call>
    <tool_call>{"name": "subtract", "arguments": {"a": 10, "b": 4}}</tool_call>
    <tool_call>{"name": "multiply", "arguments": {"a": 6, "b": 7}}</tool_call>
    <tool_call>{"name": "divide", "arguments": {"a": 20, "b": 4}}</tool_call>

The model's action stop token is </tool_call>.  The server returns a
complete <tool_response>...</tool_response> block as the observation.

Deploy by copying this file to:
    <verl_tool_root>/servers/tools/arithmetic.py
"""

import json
import re

from verl_tool.servers.tools.base import BaseTool, register_tool

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_SUPPORTED = {"add", "subtract", "multiply", "divide"}


@register_tool("arithmetic")
class ArithmeticTool(BaseTool):
    """Simple four-operation arithmetic tool (OpenAI tool-call format)."""

    tool_type = "arithmetic"

    def get_usage_inst(self) -> str:
        return (
            "Arithmetic tool — supported functions:\n"
            "  add, subtract, multiply, divide\n\n"
            'Format: <tool_call>{"name": "add", "arguments": {"a": 3, "b": 5}}</tool_call>\n'
            "The action stop token is </tool_call>; the server returns\n"
            "<tool_response>...</tool_response> as the observation."
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def parse_action(self, action: str):
        """Extract function name and arguments from an OpenAI-format tool call."""
        match = _TOOL_CALL_RE.search(action)
        if not match:
            return None, False

        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None, False

        func = payload.get("name", "").lower()
        if func not in _SUPPORTED:
            return None, False

        args = payload.get("arguments", {})
        try:
            a = float(args["a"])
            b = float(args["b"])
        except (KeyError, TypeError, ValueError):
            return None, False

        return {"func": func, "a": a, "b": b}, True

    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """Return 1 if we can handle this action, -1 otherwise."""
        _, valid = self.parse_action(action)
        return 1 if valid else -1

    def conduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        """Execute the arithmetic operation and return the observation."""
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            obs = (
                "\n<tool_response>\n"
                'Error: invalid tool call. Expected e.g. <tool_call>{"name": "add", "arguments": {"a": 3, "b": 5}}</tool_call>\n'
                "</tool_response>\n"
            )
            self.update_env(trajectory_id, env, parsed, False, extra_field, obs)
            self.save_env(trajectory_id, env)
            return obs, False, False

        func, a, b = parsed["func"], parsed["a"], parsed["b"]

        try:
            if func == "add":
                result = a + b
            elif func == "subtract":
                result = a - b
            elif func == "multiply":
                result = a * b
            elif func == "divide":
                if b == 0:
                    obs = "\n<tool_response>\nError: division by zero.\n</tool_response>\n"
                    self.update_env(trajectory_id, env, parsed, True, extra_field, obs)
                    self.save_env(trajectory_id, env)
                    return obs, False, True
                result = a / b
            else:
                raise ValueError(f"Unknown function: {func!r}")
        except Exception as exc:
            obs = f"\n<tool_response>\nError: {exc}\n</tool_response>\n"
            self.update_env(trajectory_id, env, parsed, True, extra_field, obs)
            self.save_env(trajectory_id, env)
            return obs, False, False

        # Emit an integer when the result is exact (e.g. 8 not 8.0)
        result_str = str(int(result)) if result == int(result) else f"{result:.6g}"

        # Return a complete <tool_response> block — the model did NOT write it
        obs = f"\n<tool_response>\n{result_str}\n</tool_response>\n"

        self.update_env(trajectory_id, env, parsed, True, extra_field, obs)
        self.save_env(trajectory_id, env)
        # done=False: trajectory continues; model reads the result and answers
        return obs, False, True
