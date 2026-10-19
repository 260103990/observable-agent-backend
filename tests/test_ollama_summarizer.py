import pytest

from app.agent import ollama_summarizer


def test_parse_summary_answer_returns_content():
    from app.agent.ollama_summarizer import parse_summary_answer

    data = {
        "message": {
            "role": "assistant",
            "content": "17 加 25 的结果是 42。",
        },
        "done": True,
    }

    answer = parse_summary_answer(data)

    assert answer == "17 加 25 的结果是 42。"


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\t\n"],
)
def test_parse_summary_answer_rejects_blank_content(content):
    data = {
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }

    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"message": None},
        {"message": []},
        {"message": {}},
        {"message": {"content": None}},
        {"message": {"content": 42}},
        {"message": {"content": False}},
        {"message": {"content": []}},
        {"message": {"content": {}}},
    ],
    ids=[
        "root-not-object",
        "missing-message",
        "null-message",
        "message-not-object",
        "missing-content",
        "null-content",
        "number-content",
        "boolean-content",
        "array-content",
        "object-content",
    ],
)
def test_parse_summary_answer_rejects_invalid_response(data):
    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)


def test_parse_summary_answer_accepts_empty_tool_calls():
    data = {
        "message": {
            "role": "assistant",
            "content": "结果是 42。",
            "tool_calls": [],
        },
        "done": True,
    }

    answer = ollama_summarizer.parse_summary_answer(data)

    assert answer == "结果是 42。"


def test_parse_summary_answer_rejects_tool_calls():
    data = {
        "message": {
            "role": "assistant",
            "content": "我还需要调用工具。",
            "tool_calls": [
                {
                    "function": {
                        "name": "add_numbers",
                        "arguments": {
                            "a": 17,
                            "b": 25,
                        },
                    },
                },
            ],
        },
        "done": True,
    }

    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)
