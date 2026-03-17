"""
Parallel summarization using multiple Groq API keys.
Splits remaining papers between workers, each with its own API key.

Usage:
    python scripts/parallel_summarize.py
"""
import os
import sys
import pickle
import glob
import time
import threading
import requests
import json
import networkx as nx
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Config ---
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.1-8b-instant"

API_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    raise RuntimeError("Set GROQ_API_KEYS env var (comma-separated) before running this script.")

SECTION_SUMMARY_PROMPT = (
    "Summarize the following research paper section in 2-3 concise sentences. "
    "Focus on the key findings, methods, or claims.\n\n"
    "Section text:\n{text}\n\nSummary:"
)

TOPIC_SUMMARY_PROMPT = (
    "Summarize the following group of research paper sections in 2-3 concise sentences. "
    "Describe the overarching theme and key points.\n\n"
    "Sections:\n{text}\n\nTopic summary:"
)

# Thread-safe lock for file I/O
file_lock = threading.Lock()


def llm_summarize(text, prompt_template, api_key, max_input_chars=1500):
    """Call LLM with a specific API key, with exponential backoff."""
    truncated = text[:max_input_chars]
    prompt = prompt_template.format(text=truncated)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150, "temperature": 0.3
    }
    for attempt in range(8):
        try:
            r = requests.post(LLM_URL, headers=headers, json=payload, timeout=180)
            if r.status_code == 429:
                wait = min(30 * (2 ** attempt), 120)
                print(f"    [Key ...{api_key[-6:]}] Rate limited, waiting {wait}s (attempt {attempt+1}/8)")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 7:
                wait = min(30 * (2 ** attempt), 120)
                print(f"    [Key ...{api_key[-6:]}] Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [Key ...{api_key[-6:]}] Giving up: {e}")
                return ''
    return ''


def load_tree(arxiv_id):
    path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_tree(arxiv_id, G):
    with file_lock:
        with open(os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle'), 'wb') as f:
            pickle.dump(G, f)


def get_papers_needing_section_summaries():
    """Get list of arxiv_ids that still need section summaries."""
    needs = []
    for fname in sorted(os.listdir(TREE_DIR)):
        if not fname.endswith('_tree.gpickle'):
            continue
        arxiv_id = fname.replace('_tree.gpickle', '')
        G = load_tree(arxiv_id)
        if G is None:
            continue
        sections = [n for n in G.nodes
                    if G.nodes[n].get('type') == 'section' and not G.nodes[n].get('summary', '')]
        if sections:
            needs.append(arxiv_id)
    return needs


def get_papers_needing_topic_summaries():
    """Get list of arxiv_ids that still need topic summaries."""
    needs = []
    for fname in sorted(os.listdir(TREE_DIR)):
        if not fname.endswith('_tree.gpickle'):
            continue
        arxiv_id = fname.replace('_tree.gpickle', '')
        G = load_tree(arxiv_id)
        if G is None:
            continue
        topics = [n for n in G.nodes
                  if G.nodes[n].get('type') == 'topic' and not G.nodes[n].get('summary', '')]
        if topics:
            needs.append(arxiv_id)
    return needs


def worker_section_summaries(paper_list, api_key, worker_id):
    """Process section summaries for a list of papers using one API key."""
    stats = {'summarized': 0, 'failed': 0}
    for i, arxiv_id in enumerate(paper_list):
        G = load_tree(arxiv_id)
        if G is None:
            continue
        sections = [n for n in G.nodes
                    if G.nodes[n].get('type') == 'section' and not G.nodes[n].get('summary', '')]
        if not sections:
            continue

        for section_node in sections:
            texts = [G.nodes[child]['text'] for child in G.successors(section_node)
                     if G.nodes[child].get('type') == 'chunk']
            if not texts:
                stats['failed'] += 1
                continue
            full_text = ' '.join(texts)
            summary = llm_summarize(full_text, SECTION_SUMMARY_PROMPT, api_key)
            if summary:
                G.nodes[section_node]['summary'] = summary
                stats['summarized'] += 1
            else:
                stats['failed'] += 1
            time.sleep(10)

        save_tree(arxiv_id, G)
        print(f"  [Worker {worker_id}] Paper {i+1}/{len(paper_list)} done: {arxiv_id} "
              f"(total: {stats['summarized']} summaries)")

    print(f"\n  [Worker {worker_id}] FINISHED sections - Summarized: {stats['summarized']}, Failed: {stats['failed']}")
    return stats


def worker_topic_summaries(paper_list, api_key, worker_id):
    """Process topic summaries for a list of papers using one API key."""
    stats = {'summarized': 0, 'failed': 0}
    for i, arxiv_id in enumerate(paper_list):
        G = load_tree(arxiv_id)
        if G is None:
            continue
        topics = [n for n in G.nodes
                  if G.nodes[n].get('type') == 'topic' and not G.nodes[n].get('summary', '')]
        if not topics:
            continue

        for topic_node in topics:
            section_summaries = []
            for sec in G.successors(topic_node):
                if G.nodes[sec].get('type') == 'section':
                    s = G.nodes[sec].get('summary', '')
                    title = G.nodes[sec].get('title', '')
                    if s:
                        section_summaries.append(f"[{title}] {s}")
                    elif title:
                        section_summaries.append(f"[{title}]")
            if not section_summaries:
                stats['failed'] += 1
                continue
            combined = '\n'.join(section_summaries)
            summary = llm_summarize(combined, TOPIC_SUMMARY_PROMPT, api_key)
            if summary:
                G.nodes[topic_node]['summary'] = summary
                stats['summarized'] += 1
            else:
                stats['failed'] += 1
            time.sleep(10)

        save_tree(arxiv_id, G)
        print(f"  [Worker {worker_id}] Topic paper {i+1}/{len(paper_list)} done: {arxiv_id}")

    print(f"\n  [Worker {worker_id}] FINISHED topics - Summarized: {stats['summarized']}, Failed: {stats['failed']}")
    return stats


def main():
    print("=" * 60)
    print("PARALLEL RAPTOR SUMMARIZATION")
    print(f"Using {len(API_KEYS)} API keys")
    print("=" * 60)

    # --- Phase 1: Section Summaries ---
    papers = get_papers_needing_section_summaries()
    print(f"\nPhase 1: {len(papers)} papers need section summaries")

    if papers:
        # Use single worker — all keys share same org quota (6K tokens/min)
        start = time.time()
        key = API_KEYS[0]
        print(f"  Worker 1: {len(papers)} papers, key ...{key[-6:]}")
        worker_section_summaries(papers, key, 1)
        elapsed = (time.time() - start) / 60
        print(f"\n--- Phase 1 COMPLETE in {elapsed:.1f} min: All section summaries done ---")
    else:
        print("  All section summaries already done!")

    # --- Phase 2: Topic Summaries ---
    papers = get_papers_needing_topic_summaries()
    print(f"\nPhase 2: {len(papers)} papers need topic summaries")

    if papers:
        start = time.time()
        key = API_KEYS[0]
        print(f"  Worker 1: {len(papers)} papers, key ...{key[-6:]}")
        worker_topic_summaries(papers, key, 1)
        elapsed = (time.time() - start) / 60
        print(f"\n--- Phase 2 COMPLETE in {elapsed:.1f} min: All topic summaries done ---")
    else:
        print("  All topic summaries already done!")

    print("\n" + "=" * 60)
    print("ALL SUMMARIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
