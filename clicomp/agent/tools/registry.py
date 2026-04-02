"""Tool registry for dynamic tool management."""

from typing import Any

from loguru import logger

from clicomp.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            params = tool.cast_params(params)

            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT

            if name in {"exec", "shell"} and isinstance(params, dict):
                command = str(params.get("command", ""))
                if self._blocked_privileged_command(command):
                    logger.warning("Blocked privileged shell command in tool {}: {}", name, command[:200])
                    return "Error: Privileged shell escalation is blocked (sudo/su/doas not allowed)" + _HINT

            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    @staticmethod
    def _blocked_privileged_command(command: str) -> bool:
        import re

        lower = command.lower()
        return bool(re.search(r"(^|[\s;|&()])(?:sudo|su|doas)(\s|$)", lower))

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
