"""Quick check: how many section summaries are populated in trees."""
import os, pickle

TREE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'paper_trees')
total_sections = 0
filled_sections = 0
topic_nodes = 0

for fname in sorted(os.listdir(TREE_DIR)):
    if not fname.endswith('.gpickle'):
        continue
    path = os.path.join(TREE_DIR, fname)
    with open(path, 'rb') as f:
        G = pickle.load(f)
    for n in G.nodes:
        ntype = G.nodes[n].get('type', '')
        if ntype == 'section':
            total_sections += 1
            if G.nodes[n].get('summary', ''):
                filled_sections += 1
        elif ntype == 'topic':
            topic_nodes += 1

print(f"Total sections: {total_sections}")
print(f"Sections with summary: {filled_sections}")
print(f"Sections without summary: {total_sections - filled_sections}")
print(f"Topic nodes found: {topic_nodes}")
