from app.tools.calculator import (
    AddNumbersInput,
    add_numbers,
)


def test_add_numbers_returns_sum():
    arguments = AddNumbersInput(a=2, b=3)

    result = add_numbers(arguments)

    assert result == 5
