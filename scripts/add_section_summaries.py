
import os
import sys
import pickle
import time
import networkx as nx
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

# Use Groq API (same as the rest of the project)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

if not LLM_API_KEY:
    print("ERROR: LLM_API_KEY not set in .env")
    sys.exit(1)


def llm_summarize(text, max_input_chars=3000):
    """Summarize text using Groq API."""
    truncated = text[:max_input_chars]
    prompt = (
        "Summarize the following research paper section in 2-3 concise sentences. "
        "Focus on the key findings, methods, or claims.\n\n"
        f"Section text:\n{truncated}\n\nSummary:"
    )
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.3
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(2)
            response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return ''


def summarize_section(G, section_node):
    """Concatenate all chunk texts under a section and summarize."""
    texts = [G.nodes[child]['text'] for child in G.successors(section_node)
             if G.nodes[child].get('type') == 'chunk']
    if not texts:
        return ''
    full_text = ' '.join(texts)
    return llm_summarize(full_text)


# Count total sections to summarize
total_sections = 0
trees_to_process = []
for fname in sorted(os.listdir(TREE_DIR)):
    if not fname.endswith('_tree.gpickle'):
        continue
    path = os.path.join(TREE_DIR, fname)
    with open(path, 'rb') as f:
        G = pickle.load(f)
    sections_needing_summary = [
        n for n in G.nodes
        if G.nodes[n].get('type') == 'section' and not G.nodes[n].get('summary', '')
    ]
    if sections_needing_summary:
        trees_to_process.append((fname, path, sections_needing_summary))
        total_sections += len(sections_needing_summary)

print(f"Sections needing summaries: {total_sections}")
print(f"Trees to process: {len(trees_to_process)}")

summarized = 0
failed = 0
for fname, path, sections in tqdm(trees_to_process, desc='Processing trees'):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    for section_node in sections:
        summary = summarize_section(G, section_node)
        if summary:
            G.nodes[section_node]['summary'] = summary
            summarized += 1
        else:
            failed += 1
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    with open(path, 'wb') as f:
        pickle.dump(G, f)

print(f"\nDone! Summaries added: {summarized}, Failed: {failed}")
