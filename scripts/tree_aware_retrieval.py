import os
import pickle
import networkx as nx

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

def get_chunks_for_section(arxiv_id, section_num):
    tree_path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    with open(tree_path, 'rb') as f:
        G = pickle.load(f)
    # Find section node
    for node in G.nodes:
        n = G.nodes[node]
        if n['type'] == 'section' and n.get('section_num') == section_num:
            # Return all chunk texts under this section
            return [G.nodes[child]['text'] for child in G.successors(node) if G.nodes[child]['type'] == 'chunk']
    return []

if __name__ == "__main__":
    arxiv_id = input("Enter arXiv ID: ")
    section_num = input("Enter section number (e.g., '1', '2.1'): ")
    chunks = get_chunks_for_section(arxiv_id, section_num)
    print(f"Found {len(chunks)} chunks in section {section_num} of {arxiv_id}.")
    for c in chunks[:3]:
        print(f"Chunk: {c[:200]}...\n")
