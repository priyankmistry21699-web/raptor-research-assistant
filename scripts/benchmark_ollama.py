"""Benchmark a single Ollama summary call."""
import time, requests

start = time.time()
p = {
    'model': 'mistral:latest',
    'messages': [{'role': 'user', 'content': 'Summarize the following research paper section in 2-3 concise sentences. Focus on the key findings, methods, or contributions.\n\nSection text: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.'}],
    'max_tokens': 150,
    'temperature': 0.3
}
r = requests.post('http://localhost:11435/v1/chat/completions', json=p, timeout=180)
elapsed = time.time() - start
print(f'Status: {r.status_code}, Time: {elapsed:.1f}s')
data = r.json()
content = data['choices'][0]['message']['content']
print(f'Response: {content[:300]}')
