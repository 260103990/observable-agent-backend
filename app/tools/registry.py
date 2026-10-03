from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


class ToolNotFoundError(Exception):
    pass


class ToolAlreadyRegisteredError(Exception):
    pass


class ToolArgumentsError(Exception):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[..., object]

    def to_schema(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": (
                self.arguments_model.model_json_schema()
            ),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(
                f"Tool not found: {name}"
            ) from error

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> object:
        tool = self.get(name)

        try:
            validated_arguments = (
                tool.arguments_model.model_validate(arguments)
            )
        except ValidationError as error:
            raise ToolArgumentsError(
                f"Invalid arguments for tool: {name}"
            ) from error

        return tool.handler(validated_arguments)

    def list_schemas(self) -> list[dict[str, object]]:
        return [
            tool.to_schema()
            for tool in self._tools.values()
        ]
