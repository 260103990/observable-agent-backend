import httpx
def build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

def build_chat_payload(
    model: str,
    prompt: str,
    tool_schemas: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": schema,
            }
            for schema in tool_schemas
        ],
        "stream": False,
    }


class OllamaTimeoutError(Exception):
    """Ollama 请求超时。"""


class OllamaConnectionError(Exception):
    """无法连接 Ollama 服务。"""


class OllamaResponseError(Exception):
    """Ollama 返回了错误状态。"""


async def _post_json(
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> dict:
    owns_client = client is None

    if client is None:
        client = httpx.AsyncClient(
            timeout=timeout_seconds
        )

    try:
        response = await client.post(
            url,
            json=payload,
        )
        response.raise_for_status()

        return response.json()
    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError(
            "Ollama request timed out"
        ) from exc
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            "Ollama service is unavailable"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaResponseError(
            "Ollama returned an error response"
        ) from exc
    finally:
        if owns_client:
            await client.aclose()


async def generate(
    model: str,
    prompt: str,
    generate_url: str,
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> str:
    data = await _post_json(
        url=generate_url,
        payload=build_payload(
            model=model,
            prompt=prompt,
        ),
        timeout_seconds=timeout_seconds,
        client=client,
    )

    return data["response"]

async def chat_with_tools(
    model: str,
    prompt: str,
    tool_schemas: list[dict[str, object]],
    chat_url: str,
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> dict:
    return await _post_json(
        url=chat_url,
        payload=build_chat_payload(
            model=model,
            prompt=prompt,
            tool_schemas=tool_schemas,
        ),
        timeout_seconds=timeout_seconds,
        client=client,
    )