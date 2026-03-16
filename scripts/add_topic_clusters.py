"""
Add topic clustering layer to RAPTOR trees.
Groups sections within each paper into topics using embedding similarity + clustering.

Tree structure changes from:
    root → section → chunk
to:
    root → topic → section → chunk
"""
import os
import sys
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

embedder = EmbeddingModel()

# Topic labels based on common research paper structure
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


def assign_topic_by_title(section_title):
    """Try to assign a topic based on section title keywords."""
    title_lower = section_title.lower().strip()
    for topic, keywords in TOPIC_LABELS.items():
        for kw in keywords:
            if kw in title_lower:
                return topic
    return None


def cluster_sections_by_embedding(section_texts, n_clusters):
    """Cluster section texts using embedding similarity."""
    if len(section_texts) <= 1:
        return [0] * len(section_texts)
    embeddings = [embedder.encode(text[:500]) for text in section_texts]
    embeddings = np.array(embeddings)
    n_clusters = min(n_clusters, len(section_texts))
    if n_clusters <= 1:
        return [0] * len(section_texts)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    return labels.tolist()


def add_topic_layer(G):
    """Add topic nodes between root and sections in the tree."""
    # Get all section nodes
    sections = [n for n in G.nodes if G.nodes[n].get('type') == 'section']
    if not sections:
        return G  # No sections to cluster

    # Step 1: Try title-based topic assignment first
    topic_assignments = {}
    unassigned = []
    for sec in sections:
        title = G.nodes[sec].get('title', '')
        topic = assign_topic_by_title(title)
        if topic:
            topic_assignments[sec] = topic
        else:
            unassigned.append(sec)

    # Step 2: For unassigned sections, use embedding clustering
    if unassigned:
        # Gather text for each unassigned section (summary or chunk texts)
        section_texts = []
        for sec in unassigned:
            summary = G.nodes[sec].get('summary', '')
            if summary:
                section_texts.append(summary)
            else:
                chunks = [G.nodes[c].get('text', '') for c in G.successors(sec)
                          if G.nodes[c].get('type') == 'chunk']
                section_texts.append(' '.join(chunks)[:500] if chunks else sec)

        # Determine reasonable number of clusters (max 5, or fewer)
        n_topics_for_unassigned = min(3, len(unassigned))
        labels = cluster_sections_by_embedding(section_texts, n_topics_for_unassigned)

        # Map cluster labels to topic names
        cluster_topic_names = ['topic_group_1', 'topic_group_2', 'topic_group_3']
        for i, sec in enumerate(unassigned):
            topic_assignments[sec] = cluster_topic_names[labels[i]]

    # Step 3: Create topic nodes and rewire the graph
    # Group sections by topic
    topics = {}
    for sec, topic in topic_assignments.items():
        topics.setdefault(topic, []).append(sec)

    # Add topic nodes
    for topic_name, section_list in topics.items():
        topic_node = f'topic_{topic_name}'
        # Combine section summaries for the topic summary
        summaries = [G.nodes[s].get('summary', '') for s in section_list]
        topic_summary = ' '.join(s for s in summaries if s)[:500]

        G.add_node(topic_node, type='topic', title=topic_name.replace('_', ' ').title(),
                   summary=topic_summary, section_count=len(section_list))

        # Connect root → topic
        G.add_edge('root', topic_node)

        # Rewire: remove root → section edges, add topic → section edges
        for sec in section_list:
            if G.has_edge('root', sec):
                G.remove_edge('root', sec)
            G.add_edge(topic_node, sec)

    # Handle orphan chunks directly under root — put under a 'misc' topic
    orphan_chunks = [n for n in G.successors('root')
                     if G.nodes[n].get('type') == 'chunk']
    if orphan_chunks:
        misc_topic = 'topic_uncategorized'
        if not G.has_node(misc_topic):
            G.add_node(misc_topic, type='topic', title='Uncategorized',
                       summary='', section_count=0)
            G.add_edge('root', misc_topic)
        for chunk in orphan_chunks:
            G.remove_edge('root', chunk)
            G.add_edge(misc_topic, chunk)

    return G


# Process all trees
stats = {'processed': 0, 'topics_created': 0, 'sections_assigned': 0}
for fname in tqdm(sorted(os.listdir(TREE_DIR)), desc='Adding topic layer'):
    if not fname.endswith('_tree.gpickle'):
        continue
    path = os.path.join(TREE_DIR, fname)
    with open(path, 'rb') as f:
        G = pickle.load(f)

    # Skip if topic layer already exists
    existing_topics = [n for n in G.nodes if G.nodes[n].get('type') == 'topic']
    if existing_topics:
        continue

    G = add_topic_layer(G)

    # Count
    topics = [n for n in G.nodes if G.nodes[n].get('type') == 'topic']
    sections = [n for n in G.nodes if G.nodes[n].get('type') == 'section']
    stats['processed'] += 1
    stats['topics_created'] += len(topics)
    stats['sections_assigned'] += len(sections)

    with open(path, 'wb') as f:
        pickle.dump(G, f)

print(f"\nDone!")
print(f"Trees processed: {stats['processed']}")
print(f"Topic nodes created: {stats['topics_created']}")
print(f"Sections assigned to topics: {stats['sections_assigned']}")
