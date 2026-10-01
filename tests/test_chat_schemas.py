def test_chat_request_strips_surrounding_whitespace():
    from app.schemas.chat import ChatRequest

    request = ChatRequest(
        message="  Hello world  ",
    )

    assert request.message == "Hello world"
