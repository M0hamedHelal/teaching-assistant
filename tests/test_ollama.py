# import requests

# response = requests.post(
#     "http://localhost:11434/api/generate",
#     json={
#         "model": "llama3.2",
#         "prompt": "Hello",
#         "stream": False
#     }
# )

# print(response.json()["response"])
# tests/test_ollama.py
import requests
import pytest

OLLAMA_URL = "http://localhost:11434/api/generate"


def _ollama_is_running() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


@pytest.mark.skipif(not _ollama_is_running(), reason="Ollama server is not running")
def test_ollama_responds():
    response = requests.post(
        OLLAMA_URL,
        json={"model": "llama3.2", "prompt": "Hello", "stream": False},
        timeout=30,
    )
    assert response.status_code == 200
    assert "response" in response.json()