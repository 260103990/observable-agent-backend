from collections.abc import Awaitable, Callable

from app.schemas.agent import ToolCall
from app.tools.registry import ToolRegistry


DecisionMaker = Callable[
    [
        str,
        list[dict[str, object]],
    ],
    Awaitable[ToolCall],
]


class AgentRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        decision_maker: DecisionMaker,
    ) -> None:
        self._registry = registry
        self._decision_maker = decision_maker

    async def run(self, prompt: str) -> object:
        tool_schemas = self._registry.list_schemas()

        tool_call = await self._decision_maker(
            prompt,
            tool_schemas,
        )

        return self._registry.execute(
            tool_call.name,
            tool_call.arguments,
        )
