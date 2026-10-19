class OllamaSummaryError(Exception):
    pass


def parse_summary_answer(data: object) -> str:
    if not isinstance(data, dict):
        raise OllamaSummaryError(
            "Ollama returned an invalid summary response"
        )

    message = data.get("message")

    if not isinstance(message, dict):
        raise OllamaSummaryError(
            "Ollama returned an invalid summary message"
        )
    if message.get("tool_calls"):
        raise OllamaSummaryError(
            "Ollama returned tool calls instead of a final answer"
        )

    content = message.get("content")

    if not isinstance(content, str):
        raise OllamaSummaryError(
            "Ollama summary content must be a string"
        )

    if not content.strip():
        raise OllamaSummaryError(
            "Ollama returned an empty summary"
        )

    return content
