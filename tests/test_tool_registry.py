import pytest

from app.tools.calculator import ADD_NUMBERS_TOOL
from app.tools.registry import (
    ToolAlreadyRegisteredError,
    ToolArgumentsError,
    ToolNotFoundError,
    ToolRegistry,
)


def test_registry_registers_and_returns_tool():
    registry = ToolRegistry()

    registry.register(ADD_NUMBERS_TOOL)

    registered_tool = registry.get("add_numbers")

    assert registered_tool is ADD_NUMBERS_TOOL


def test_registry_executes_registered_tool():
    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    result = registry.execute(
        "add_numbers",
        {
            "a": 2,
            "b": 3,
        },
    )

    assert result == 5


def test_registry_rejects_unknown_tool_name():
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="missing_tool",
    ):
        registry.get("missing_tool")


def test_registry_rejects_duplicate_tool_name():
    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    with pytest.raises(
        ToolAlreadyRegisteredError,
        match="add_numbers",
    ):
        registry.register(ADD_NUMBERS_TOOL)


def test_registry_rejects_invalid_tool_arguments():
    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    with pytest.raises(
        ToolArgumentsError,
        match="add_numbers",
    ):
        registry.execute(
            "add_numbers",
            {
                "a": "not-a-number",
                "b": 3,
            },
        )


def test_tool_definition_exports_json_schema():
    schema = ADD_NUMBERS_TOOL.to_schema()

    assert schema["name"] == "add_numbers"
    assert schema["description"] == "Add two integers."

    parameters = schema["parameters"]

    assert isinstance(parameters, dict)
    assert parameters["type"] == "object"
    assert parameters["properties"]["a"]["type"] == "integer"
    assert parameters["properties"]["b"]["type"] == "integer"
    assert set(parameters["required"]) == {"a", "b"}


def test_registry_lists_registered_tool_schemas():
    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    schemas = registry.list_schemas()

    assert schemas == [
        ADD_NUMBERS_TOOL.to_schema(),
    ]
