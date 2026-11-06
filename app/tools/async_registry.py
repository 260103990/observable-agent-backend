"""Async-only tool execution with validated arguments and opt-in safe retries."""
import asyncio
from dataclasses import dataclass
import inspect
import json
import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError


class ToolTransientError(Exception):
    """A handler explicitly classifies this failure as transient."""


@dataclass(frozen=True)
class PreparedToolResult:
    """Commit local evidence state only after the executor accepts its JSON payload."""
    output: object
    accept: Callable[[], None]


@dataclass(frozen=True)
class AsyncTool:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: object
    read_only: bool = False
    max_attempts: int = 1

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.name):
            raise ValueError('Invalid tool name')
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError('Tool description is required')
        if not isinstance(self.arguments_model, type) or not issubclass(self.arguments_model, BaseModel):
            raise ValueError('Tool arguments require a Pydantic model')
        if not inspect.iscoroutinefunction(self.handler):
            raise ValueError('Tool handler must be an async function')
        if type(self.read_only) is not bool or type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 3:
            raise ValueError('Invalid retry policy')
        if not self.read_only and self.max_attempts != 1:
            raise ValueError('Side-effecting tools cannot enable automatic retries')

    def schema(self):
        return {'type':'function', 'function':{
            'name':self.name, 'description':self.description,
            'parameters':self.arguments_model.model_json_schema(),
        }}


class AsyncToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: AsyncTool):
        if tool.name in self._tools:
            raise ValueError('Tool is already registered')
        self._tools[tool.name] = tool

    def list_schemas(self):
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, name, arguments, *, timeout_seconds, max_result_chars):
        def failure(code, attempts):
            return {'ok':False, 'error':code, 'attempts':attempts}
        tool = self._tools.get(name)
        if tool is None:
            return failure('unknown_tool', 0)
        try:
            validated = tool.arguments_model.model_validate(arguments, strict=True, extra='forbid')
        except ValidationError:
            return failure('invalid_arguments', 0)
        for attempt in range(1, tool.max_attempts + 1):
            try:
                async with asyncio.timeout(timeout_seconds):
                    output = await tool.handler(validated)
            except ToolTransientError:
                if attempt < tool.max_attempts:
                    continue
                return failure('tool_transient_failure', attempt)
            except TimeoutError:
                return failure('tool_timeout', attempt)
            except Exception:
                return failure('tool_failed', attempt)
            prepared = output if isinstance(output, PreparedToolResult) else None
            result = {'ok':True, 'output':prepared.output if prepared else output, 'attempts':attempt}
            try:
                encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
            except (ValueError, TypeError, OverflowError):
                return failure('invalid_tool_result', attempt)
            if len(encoded) > max_result_chars:
                return failure('tool_result_too_large', attempt)
            if prepared:
                try:
                    prepared.accept()
                except Exception:
                    return failure('tool_result_commit_failed', attempt)
            return result
