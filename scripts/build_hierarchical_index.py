
import os
import re
import json
import pickle
import networkx as nx
import fitz  # PyMuPDF
from tqdm import tqdm

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CHUNKS_PATH = os.path.join(RAW_DATA_DIR, 'papers_chunks.jsonl')
METADATA_PATH = os.path.join(RAW_DATA_DIR, 'papers_metadata_with_id.json')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

os.makedirs(TREE_DIR, exist_ok=True)

# --- Improved section header detection ---
# Matches numbered sections: "1 Introduction", "2.1 Related Work", "3. Methods"
NUMBERED_SECTION_RE = re.compile(
    r'^(\d+(?:\.\d+)*)[.\s\-:]+([A-Z][A-Za-z\s\-&,]+)', re.MULTILINE
)
# Matches common named sections without numbers
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


def extract_title_from_pdf(pdf_path):
    """Fallback: extract paper title from the first page of the PDF."""
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0].get_text()
        lines = [l.strip() for l in first_page.split('\n') if l.strip()]
        # Heuristic: title is usually the first non-empty, long-ish line
        for line in lines[:5]:
            if len(line) > 10 and not line.startswith('arXiv'):
                return line
    except Exception:
        pass
    return ''


def detect_section(text):
    """Try to detect a section header in the chunk text.
    Returns (section_num, section_title) or None."""
    # Try numbered section first
    m = NUMBERED_SECTION_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    # Also search within the first 200 chars (header may not be at pos 0)
    m = NUMBERED_SECTION_RE.search(text[:200])
    if m:
        return m.group(1), m.group(2).strip()
    # Try named sections
    m = NAMED_SECTION_RE.match(text)
    if m:
        return m.group(1).lower().replace(' ', '_'), m.group(1).strip()
    m = NAMED_SECTION_RE.search(text[:200])
    if m:
        return m.group(1).lower().replace(' ', '_'), m.group(1).strip()
    return None


# Load metadata
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata_lookup = {p['arxiv_id']: p for p in json.load(f)}

# Group chunks by paper
papers = {}
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        papers.setdefault(rec['arxiv_id'], []).append(rec)

stats = {'with_meta': 0, 'fallback_title': 0, 'sections_found': 0, 'total_chunks': 0}

for arxiv_id, chunks in tqdm(papers.items(), desc='Building trees'):
    meta = metadata_lookup.get(arxiv_id, {})
    title = meta.get('title', '')

    # Fallback: extract title from PDF if metadata is missing
    if not title:
        pdf_path = os.path.join(RAW_DATA_DIR, f'{arxiv_id}.pdf')
        if os.path.exists(pdf_path):
            title = extract_title_from_pdf(pdf_path)
            if title:
                stats['fallback_title'] += 1
    else:
        stats['with_meta'] += 1

    G = nx.DiGraph()
    # Add root node (paper)
    G.add_node('root', type='paper', arxiv_id=arxiv_id, title=title, metadata=meta)

    # --- Two-pass approach ---
    # Pass 1: scan all chunks to find section boundaries
    section_assignments = []  # (chunk, section_id_or_None)
    section_nodes = {}

    for chunk in chunks:
        text = chunk['text']
        result = detect_section(text)
        if result:
            section_num, section_title = result
            section_id = f'section_{section_num}'
            if section_id not in section_nodes:
                G.add_node(section_id, type='section', title=section_title,
                           section_num=section_num, summary='')
                G.add_edge('root', section_id)
                section_nodes[section_id] = section_title
                stats['sections_found'] += 1
            section_assignments.append((chunk, section_id))
        else:
            section_assignments.append((chunk, None))

    # Pass 2: assign chunks to sections (propagate last known section)
    current_section = 'root'
    for chunk, detected_section in section_assignments:
        if detected_section:
            current_section = detected_section
        chunk_id = f"chunk_{chunk['chunk_index']}"
        G.add_node(chunk_id, type='chunk', text=chunk['text'],
                   chunk_index=chunk['chunk_index'])
        G.add_edge(current_section, chunk_id)
        stats['total_chunks'] += 1

    # Save tree
    with open(os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle'), 'wb') as f:
        pickle.dump(G, f)

print(f"\nSaved all paper trees to {TREE_DIR}")
print(f"Papers with metadata: {stats['with_meta']}")
print(f"Papers with fallback title from PDF: {stats['fallback_title']}")
print(f"Total sections detected: {stats['sections_found']}")
print(f"Total chunks assigned: {stats['total_chunks']}")
