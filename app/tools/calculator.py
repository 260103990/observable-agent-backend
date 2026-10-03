from pydantic import BaseModel

from app.tools.registry import ToolDefinition


class AddNumbersInput(BaseModel):
    a: int
    b: int


def add_numbers(arguments: AddNumbersInput) -> int:
    return arguments.a + arguments.b


ADD_NUMBERS_TOOL = ToolDefinition(
    name="add_numbers",
    description="Add two integers.",
    arguments_model=AddNumbersInput,
    handler=add_numbers,
)
