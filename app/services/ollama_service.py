import httpx
def build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }


class OllamaTimeoutError(Exception):
    """Ollama 请求超时。"""


class OllamaConnectionError(Exception):
    """无法连接 Ollama 服务。"""


class OllamaResponseError(Exception):
    """Ollama 返回了错误状态。"""


async def generate(
    model: str,
    prompt: str,
    generate_url: str,
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> str:
    owns_client = client is None

    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)

    try:
        response = await client.post(
            generate_url,
            json=build_payload(model=model, prompt=prompt),
        )
        response.raise_for_status()

        data = response.json()
        return data["response"]
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