import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": False
    }
)

print(response.json()["response"])