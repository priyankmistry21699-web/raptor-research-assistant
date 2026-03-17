"""
Unified RAPTOR Index Builder

Runs the full pipeline to build 4-level hierarchical trees:
    1. Build base trees (root → section → chunk) from chunks + metadata
    2. Generate section summaries via LLM
    3. Add topic clustering layer (root → topic → section → chunk)
    4. Generate topic summaries via LLM

Usage:
    python scripts/build_index.py                  # full pipeline
    python scripts/build_index.py --step trees      # only build base trees
    python scripts/build_index.py --step summaries  # only add section summaries
    python scripts/build_index.py --step topics     # only add topic layer
    python scripts/build_index.py --step topic-summaries  # only add topic summaries
    python scripts/build_index.py --paper 1706.03762  # single paper only
"""
import os
import sys
import re
import json
import pickle
import time
import argparse
import numpy as np
import networkx as nx
import fitz  # PyMuPDF
import requests
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Paths ---
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CHUNKS_PATH = os.path.join(RAW_DATA_DIR, 'papers_chunks.jsonl')
METADATA_PATH = os.path.join(RAW_DATA_DIR, 'papers_metadata_with_id.json')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')
os.makedirs(TREE_DIR, exist_ok=True)

# --- LLM Config ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# --- Section Detection ---
NUMBERED_SECTION_RE = re.compile(
    r'^(\d+(?:\.\d+)*)[.\s\-:]+([A-Z][A-Za-z\s\-&,]+)', re.MULTILINE
)
NAMED_SECTIONS = [
    'Abstract', 'Introduction', 'Related Work', 'Background',
    'Methodology', 'Methods', 'Method', 'Approach',
    'Experiments', 'Experimental Setup', 'Experimental Results',
    'Results', 'Results and Discussion', 'Discussion',
    'Analysis', 'Evaluation', 'Ablation Study', 'Ablation',
    'Conclusion', 'Conclusions', 'Conclusion and Future Work',
    'Future Work', 'Limitations', 'Acknowledgements', 'Acknowledgments',
    'References', 'Appendix', 'Supplementary Material',
]
NAMED_SECTION_RE = re.compile(
    r'^(' + '|'.join(re.escape(s) for s in NAMED_SECTIONS) + r')\s*$',
    re.MULTILINE | re.IGNORECASE
)

# --- Topic Labels ---
TOPIC_LABELS = {
    'background': ['abstract', 'introduction', 'background', 'related_work', 'related work',
                    'motivation', 'overview', 'preliminaries', 'problem_statement'],
    'methodology': ['method', 'methods', 'methodology', 'approach', 'model', 'architecture',
                     'framework', 'algorithm', 'design', 'implementation', 'proposed'],
    'experiments': ['experiment', 'experiments', 'experimental_setup', 'experimental setup',
                     'experimental_results', 'setup', 'datasets', 'training', 'evaluation'],
    'results': ['results', 'results_and_discussion', 'results and discussion', 'analysis',
                'ablation', 'ablation_study', 'comparison', 'performance', 'findings'],
    'conclusion': ['conclusion', 'conclusions', 'conclusion_and_future_work',
                    'future_work', 'future work', 'limitations', 'discussion',
                    'summary', 'acknowledgements', 'acknowledgments', 'references', 'appendix'],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_section(text):
    """Detect section header in chunk text. Returns (section_num, section_title) or None."""
    m = NUMBERED_SECTION_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    m = NUMBERED_SECTION_RE.search(text[:200])
    if m:
        return m.group(1), m.group(2).strip()
    m = NAMED_SECTION_RE.match(text)
    if m:
        return m.group(1).lower().replace(' ', '_'), m.group(1).strip()
    m = NAMED_SECTION_RE.search(text[:200])
    if m:
        return m.group(1).lower().replace(' ', '_'), m.group(1).strip()
    return None


def extract_title_from_pdf(pdf_path):
    """Fallback: extract paper title from the first page of the PDF."""
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0].get_text()
        lines = [l.strip() for l in first_page.split('\n') if l.strip()]
        for line in lines[:5]:
            if len(line) > 10 and not line.startswith('arXiv'):
                return line
    except Exception:
        pass
    return ''


def llm_summarize(text, prompt_template, max_input_chars=3000):
    """Call LLM to generate a summary with exponential backoff for rate limits."""
    if not LLM_API_KEY:
        return ''
    truncated = text[:max_input_chars]
    prompt = prompt_template.format(text=truncated)
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
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=180)
            if response.status_code == 429:
                wait = min(30 * (2 ** attempt), 120)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(30 * (2 ** attempt), 120)
                print(f"  LLM error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  LLM error (giving up): {e}")
                return ''
    return ''


def assign_topic_by_title(section_title):
    """Assign a topic based on section title keywords."""
    title_lower = section_title.lower().strip()
    for topic, keywords in TOPIC_LABELS.items():
        for kw in keywords:
            if kw in title_lower:
                return topic
    return None


def cluster_sections_by_embedding(embedder, section_texts, n_clusters):
    """Cluster section texts using embedding similarity."""
    if len(section_texts) <= 1:
        return [0] * len(section_texts)
    embeddings = [embedder.encode(text[:500]) for text in section_texts]
    embeddings = np.array(embeddings)
    n_clusters = min(n_clusters, len(section_texts))
    if n_clusters <= 1:
        return [0] * len(section_texts)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric='cosine', linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    return labels.tolist()


def save_tree(arxiv_id, G):
    """Save a tree to disk."""
    with open(os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle'), 'wb') as f:
        pickle.dump(G, f)


def load_tree(arxiv_id):
    """Load a tree from disk."""
    path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================
# STEP 1: Build Base Trees (root → section → chunk)
# ============================================================

def step_build_trees(paper_filter=None):
    """Build 3-level trees: root → section → chunk."""
    print("\n=== STEP 1: Building base trees (root → section → chunk) ===")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata_lookup = {p['arxiv_id']: p for p in json.load(f)}

    papers = {}
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            papers.setdefault(rec['arxiv_id'], []).append(rec)

    if paper_filter:
        papers = {k: v for k, v in papers.items() if k == paper_filter}

    stats = {'papers': 0, 'sections': 0, 'chunks': 0}

    for arxiv_id, chunks in tqdm(papers.items(), desc='Building trees'):
        meta = metadata_lookup.get(arxiv_id, {})
        title = meta.get('title', '')
        if not title:
            pdf_path = os.path.join(RAW_DATA_DIR, f'{arxiv_id}.pdf')
            if os.path.exists(pdf_path):
                title = extract_title_from_pdf(pdf_path)

        G = nx.DiGraph()
        G.add_node('root', type='paper', arxiv_id=arxiv_id, title=title, metadata=meta)

        # Pass 1: detect section boundaries
        section_assignments = []
        section_nodes = {}
        for chunk in chunks:
            result = detect_section(chunk['text'])
            if result:
                section_num, section_title = result
                section_id = f'section_{section_num}'
                if section_id not in section_nodes:
                    G.add_node(section_id, type='section', title=section_title,
                               section_num=section_num, summary='')
                    G.add_edge('root', section_id)
                    section_nodes[section_id] = section_title
                    stats['sections'] += 1
                section_assignments.append((chunk, section_id))
            else:
                section_assignments.append((chunk, None))

        # Pass 2: assign chunks to sections
        current_section = 'root'
        for chunk, detected_section in section_assignments:
            if detected_section:
                current_section = detected_section
            chunk_id = f"chunk_{chunk['chunk_index']}"
            G.add_node(chunk_id, type='chunk', text=chunk['text'],
                       chunk_index=chunk['chunk_index'])
            G.add_edge(current_section, chunk_id)
            stats['chunks'] += 1

        save_tree(arxiv_id, G)
        stats['papers'] += 1

    print(f"  Trees built: {stats['papers']}, Sections: {stats['sections']}, Chunks: {stats['chunks']}")
    return stats


# ============================================================
# STEP 2: Generate Section Summaries
# ============================================================

SECTION_SUMMARY_PROMPT = (
    "Summarize the following research paper section in 2-3 concise sentences. "
    "Focus on the key findings, methods, or claims.\n\n"
    "Section text:\n{text}\n\nSummary:"
)


def step_add_section_summaries(paper_filter=None):
    """Add LLM-generated summaries to section nodes."""
    print("\n=== STEP 2: Generating section summaries ===")

    if not LLM_API_KEY:
        print("  WARNING: LLM_API_KEY not set. Skipping summaries.")
        return {'summarized': 0, 'failed': 0}

    stats = {'summarized': 0, 'failed': 0, 'skipped': 0}

    for fname in tqdm(sorted(os.listdir(TREE_DIR)), desc='Section summaries'):
        if not fname.endswith('_tree.gpickle'):
            continue
        arxiv_id = fname.replace('_tree.gpickle', '')
        if paper_filter and arxiv_id != paper_filter:
            continue

        G = load_tree(arxiv_id)
        if G is None:
            continue

        sections = [n for n in G.nodes
                    if G.nodes[n].get('type') == 'section' and not G.nodes[n].get('summary', '')]

        if not sections:
            stats['skipped'] += 1
            continue

        for section_node in sections:
            texts = [G.nodes[child]['text'] for child in G.successors(section_node)
                     if G.nodes[child].get('type') == 'chunk']
            if not texts:
                stats['failed'] += 1
                continue

            full_text = ' '.join(texts)
            summary = llm_summarize(full_text, SECTION_SUMMARY_PROMPT)
            if summary:
                G.nodes[section_node]['summary'] = summary
                stats['summarized'] += 1
            else:
                stats['failed'] += 1
            time.sleep(10)

        save_tree(arxiv_id, G)

    print(f"  Summaries added: {stats['summarized']}, Failed: {stats['failed']}, "
          f"Skipped (already done): {stats['skipped']}")
    return stats


# ============================================================
# STEP 3: Add Topic Clustering Layer
# ============================================================

def step_add_topic_layer(paper_filter=None):
    """Add topic nodes between root and sections."""
    print("\n=== STEP 3: Adding topic clustering layer ===")

    embedder = EmbeddingModel()
    stats = {'processed': 0, 'topics': 0, 'skipped': 0}

    for fname in tqdm(sorted(os.listdir(TREE_DIR)), desc='Topic clustering'):
        if not fname.endswith('_tree.gpickle'):
            continue
        arxiv_id = fname.replace('_tree.gpickle', '')
        if paper_filter and arxiv_id != paper_filter:
            continue

        G = load_tree(arxiv_id)
        if G is None:
            continue

        # Skip if topic layer already exists
        if any(G.nodes[n].get('type') == 'topic' for n in G.nodes):
            stats['skipped'] += 1
            continue

        sections = [n for n in G.nodes if G.nodes[n].get('type') == 'section']
        if not sections:
            stats['skipped'] += 1
            continue

        # Title-based topic assignment
        topic_assignments = {}
        unassigned = []
        for sec in sections:
            title = G.nodes[sec].get('title', '')
            topic = assign_topic_by_title(title)
            if topic:
                topic_assignments[sec] = topic
            else:
                unassigned.append(sec)

        # Embedding-based clustering for unassigned sections
        if unassigned:
            section_texts = []
            for sec in unassigned:
                summary = G.nodes[sec].get('summary', '')
                if summary:
                    section_texts.append(summary)
                else:
                    chunks = [G.nodes[c].get('text', '') for c in G.successors(sec)
                              if G.nodes[c].get('type') == 'chunk']
                    section_texts.append(' '.join(chunks)[:500] if chunks else sec)

            n_topics = min(3, len(unassigned))
            labels = cluster_sections_by_embedding(embedder, section_texts, n_topics)
            cluster_names = ['topic_group_1', 'topic_group_2', 'topic_group_3']
            for i, sec in enumerate(unassigned):
                topic_assignments[sec] = cluster_names[labels[i]]

        # Group sections by topic
        topics = {}
        for sec, topic in topic_assignments.items():
            topics.setdefault(topic, []).append(sec)

        # Create topic nodes and rewire
        for topic_name, section_list in topics.items():
            topic_node = f'topic_{topic_name}'
            summaries = [G.nodes[s].get('summary', '') for s in section_list]
            topic_summary = ' '.join(s for s in summaries if s)[:500]

            G.add_node(topic_node, type='topic',
                       title=topic_name.replace('_', ' ').title(),
                       summary=topic_summary,
                       section_count=len(section_list))
            G.add_edge('root', topic_node)

            for sec in section_list:
                if G.has_edge('root', sec):
                    G.remove_edge('root', sec)
                G.add_edge(topic_node, sec)

        # Handle orphan chunks directly under root
        orphan_chunks = [n for n in G.successors('root')
                         if G.nodes[n].get('type') == 'chunk']
        if orphan_chunks:
            misc = 'topic_uncategorized'
            if not G.has_node(misc):
                G.add_node(misc, type='topic', title='Uncategorized',
                           summary='', section_count=0)
                G.add_edge('root', misc)
            for chunk in orphan_chunks:
                G.remove_edge('root', chunk)
                G.add_edge(misc, chunk)

        save_tree(arxiv_id, G)
        new_topics = [n for n in G.nodes if G.nodes[n].get('type') == 'topic']
        stats['processed'] += 1
        stats['topics'] += len(new_topics)

    print(f"  Trees processed: {stats['processed']}, Topics created: {stats['topics']}, "
          f"Skipped: {stats['skipped']}")
    return stats


# ============================================================
# STEP 4: Generate Topic Summaries
# ============================================================

TOPIC_SUMMARY_PROMPT = (
    "Summarize the following group of research paper sections in 2-3 concise sentences. "
    "Describe the overarching theme and key points.\n\n"
    "Sections:\n{text}\n\nTopic summary:"
)


def step_add_topic_summaries(paper_filter=None):
    """Add LLM-generated summaries to topic nodes."""
    print("\n=== STEP 4: Generating topic summaries ===")

    if not LLM_API_KEY:
        print("  WARNING: LLM_API_KEY not set. Skipping topic summaries.")
        return {'summarized': 0, 'failed': 0}

    stats = {'summarized': 0, 'failed': 0, 'skipped': 0}

    for fname in tqdm(sorted(os.listdir(TREE_DIR)), desc='Topic summaries'):
        if not fname.endswith('_tree.gpickle'):
            continue
        arxiv_id = fname.replace('_tree.gpickle', '')
        if paper_filter and arxiv_id != paper_filter:
            continue

        G = load_tree(arxiv_id)
        if G is None:
            continue

        topics = [n for n in G.nodes
                  if G.nodes[n].get('type') == 'topic' and not G.nodes[n].get('summary', '')]

        if not topics:
            stats['skipped'] += 1
            continue

        for topic_node in topics:
            # Gather section summaries under this topic
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
            summary = llm_summarize(combined, TOPIC_SUMMARY_PROMPT)
            if summary:
                G.nodes[topic_node]['summary'] = summary
                stats['summarized'] += 1
            else:
                stats['failed'] += 1
            time.sleep(10)

        save_tree(arxiv_id, G)

    print(f"  Topic summaries added: {stats['summarized']}, Failed: {stats['failed']}, "
          f"Skipped: {stats['skipped']}")
    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Build RAPTOR hierarchical index')
    parser.add_argument('--step', choices=['trees', 'summaries', 'topics', 'topic-summaries', 'all'],
                        default='all', help='Which pipeline step to run')
    parser.add_argument('--paper', type=str, default=None,
                        help='Process a single paper by arXiv ID')
    args = parser.parse_args()

    print("=" * 60)
    print("RAPTOR Hierarchical Index Builder")
    print("=" * 60)

    steps = {
        'trees': [step_build_trees],
        'summaries': [step_add_section_summaries],
        'topics': [step_add_topic_layer],
        'topic-summaries': [step_add_topic_summaries],
        'all': [step_build_trees, step_add_section_summaries,
                step_add_topic_layer, step_add_topic_summaries],
    }

    for step_fn in steps[args.step]:
        step_fn(paper_filter=args.paper)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
