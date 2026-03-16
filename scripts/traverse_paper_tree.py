import os
import pickle
import networkx as nx

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

def print_tree(arxiv_id):
    tree_path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    with open(tree_path, 'rb') as f:
        G = pickle.load(f)
    def print_node(node, depth=0):
        n = G.nodes[node]
        prefix = '  ' * depth
        if n['type'] == 'paper':
            print(f"{prefix}Paper: {n.get('title', '')} [{n.get('arxiv_id', '')}]")
        elif n['type'] == 'section':
            print(f"{prefix}Section {n.get('section_num', '')}: {n.get('title', '')}")
            if 'summary' in n:
                print(f"{prefix}  Summary: {n['summary']}")
        elif n['type'] == 'chunk':
            print(f"{prefix}Chunk {n.get('chunk_index', '')}: {n.get('text', '')[:60]}...")
        for child in G.successors(node):
            print_node(child, depth+1)
    print_node('root')

if __name__ == "__main__":
    arxiv_id = input("Enter arXiv ID to traverse: ")
    print_tree(arxiv_id)
