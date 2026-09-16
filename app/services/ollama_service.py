import httpx
def build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


class OllamaTimeoutError(Exception):
    """Ollama 请求超时。"""


class OllamaConnectionError(Exception):
    """无法连接 Ollama 服务。"""

async def generate(
    model: str,
    prompt: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    owns_client = client is None

    if client is None:
        client = httpx.AsyncClient(timeout=120.0)

    try:
        response = await client.post(
            OLLAMA_GENERATE_URL,
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
    finally:
        if owns_client:
            await client.aclose()