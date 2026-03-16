import requests
import json

query = "What are the advantages of deep learning over traditional machine learning?"
r1 = requests.post("http://localhost:8000/retrieve", json={"query": query, "top_k": 5})
r2 = requests.post("http://localhost:8000/prompt", json={"query": query, "top_k": 5})

print("STATUS retrieve:", r1.status_code)
print("STATUS prompt:", r2.status_code)
print("Chunks retrieved:", len(r1.json()))
print()

for i, c in enumerate(r1.json()):
    title = c["paper_title"] or "(no title)"
    text_preview = c["chunk_text"][:120].replace("\n", " ")
    print(f"{i+1}. [{c['arxiv_id']}] Paper: {title}")
    print(f"   Text: {text_preview}...")
    print()

output = {
    "query": query,
    "retrieve_result": r1.json(),
    "prompt_result": r2.json()
}
with open("data/raw/test_query_result_2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Saved to data/raw/test_query_result_2.json")
