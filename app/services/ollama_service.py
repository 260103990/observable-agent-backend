import httpx
def build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


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
    finally:
        if owns_client:
            await client.aclose()