import requests

API_URL = "http://localhost:8000/llm"

payload = {
    "query": "Explain transformer models for NLP.",
    "top_k": 3,
    "model": "llama-3.3-70b-versatile"
}

response = requests.post(API_URL, json=payload, timeout=180)
response.raise_for_status()
data = response.json()

print("Prompt sent to LLM:\n", data["prompt"])
print("\nLLM Answer:\n", data["answer"])
