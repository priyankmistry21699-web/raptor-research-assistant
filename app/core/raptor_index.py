"""
RAPTOR Hierarchical Index — Core module.
Provides tree loading, traversal, and hierarchical retrieval.

Tree structure (4 levels):
    root (paper) → topic → section → chunk
"""
import os
import pickle
import networkx as nx
from typing import List, Dict, Optional, Any

TREE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'paper_trees')


def load_tree(arxiv_id: str) -> Optional[nx.DiGraph]:
    """Load a paper's RAPTOR tree from disk."""
    path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_tree(arxiv_id: str, G: nx.DiGraph) -> None:
    """Save a paper's RAPTOR tree to disk."""
    os.makedirs(TREE_DIR, exist_ok=True)
    path = os.path.join(TREE_DIR, f'{arxiv_id}_tree.gpickle')
    with open(path, 'wb') as f:
        pickle.dump(G, f)


def get_paper_info(G: nx.DiGraph) -> Dict[str, Any]:
    """Get paper-level info from the root node."""
    root = G.nodes['root']
    return {
        'arxiv_id': root.get('arxiv_id', ''),
        'title': root.get('title', ''),
        'metadata': root.get('metadata', {}),
    }


def get_topics(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Get all topic nodes from the tree."""
    topics = []
    for node in G.successors('root'):
        n = G.nodes[node]
        if n.get('type') == 'topic':
            sections = [s for s in G.successors(node) if G.nodes[s].get('type') == 'section']
            topics.append({
                'node_id': node,
                'title': n.get('title', ''),
                'summary': n.get('summary', ''),
                'section_count': len(sections),
            })
    return topics


def get_sections(G: nx.DiGraph, parent_node: str = None) -> List[Dict[str, Any]]:
    """Get sections, optionally filtered by parent topic node."""
    sections = []
    if parent_node:
        candidates = list(G.successors(parent_node))
    else:
        candidates = list(G.nodes)
    for node in candidates:
        n = G.nodes[node]
        if n.get('type') == 'section':
            chunk_count = sum(1 for c in G.successors(node) if G.nodes[c].get('type') == 'chunk')
            sections.append({
                'node_id': node,
                'section_num': n.get('section_num', ''),
                'title': n.get('title', ''),
                'summary': n.get('summary', ''),
                'chunk_count': chunk_count,
            })
    return sections


def get_chunks(G: nx.DiGraph, parent_node: str) -> List[Dict[str, Any]]:
    """Get all chunks under a given node (section or topic)."""
    chunks = []
    for child in G.successors(parent_node):
        n = G.nodes[child]
        if n.get('type') == 'chunk':
            chunks.append({
                'node_id': child,
                'chunk_index': n.get('chunk_index', 0),
                'text': n.get('text', ''),
            })
        elif n.get('type') in ('section', 'topic'):
            chunks.extend(get_chunks(G, child))
    return chunks


def get_tree_structure(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Get the full hierarchical structure for a paper."""
    G = load_tree(arxiv_id)
    if G is None:
        return None
    info = get_paper_info(G)
    topics = get_topics(G)

    # If no topics exist (old 3-level tree), show sections directly under root
    if not topics:
        sections = get_sections(G, 'root')
        structure = {
            **info,
            'topics': [],
            'sections': sections,
            'total_chunks': sum(1 for n in G.nodes if G.nodes[n].get('type') == 'chunk'),
            'total_sections': len(sections),
            'total_topics': 0,
            'tree_levels': 3,
        }
        return structure

    structure = {
        **info,
        'topics': [],
        'sections': [],
        'total_chunks': sum(1 for n in G.nodes if G.nodes[n].get('type') == 'chunk'),
        'total_sections': sum(1 for n in G.nodes if G.nodes[n].get('type') == 'section'),
        'total_topics': len(topics),
        'tree_levels': 4,
    }
    for topic in topics:
        topic_data = {**topic, 'sections': []}
        for sec in get_sections(G, topic['node_id']):
            sec_data = {**sec, 'chunk_count': sec['chunk_count']}
            topic_data['sections'].append(sec_data)
        structure['topics'].append(topic_data)
    return structure


def get_context_for_chunk(G: nx.DiGraph, chunk_node: str) -> Dict[str, Any]:
    """Given a chunk node, walk UP the tree to get full hierarchical context."""
    context = {
        'paper_title': G.nodes['root'].get('title', ''),
        'arxiv_id': G.nodes['root'].get('arxiv_id', ''),
        'topic': '',
        'topic_summary': '',
        'section_num': '',
        'section_title': '',
        'section_summary': '',
    }
    # Find parent section
    for parent in G.predecessors(chunk_node):
        n = G.nodes[parent]
        if n.get('type') == 'section':
            context['section_num'] = n.get('section_num', '')
            context['section_title'] = n.get('title', '')
            context['section_summary'] = n.get('summary', '')
            # Find parent topic
            for grandparent in G.predecessors(parent):
                gn = G.nodes[grandparent]
                if gn.get('type') == 'topic':
                    context['topic'] = gn.get('title', '')
                    context['topic_summary'] = gn.get('summary', '')
                    break
            break
        elif n.get('type') == 'topic':
            context['topic'] = n.get('title', '')
            context['topic_summary'] = n.get('summary', '')
            break
    return context


def get_tree_stats(arxiv_id: str) -> Optional[Dict[str, int]]:
    """Get node-type counts for a paper's tree."""
    G = load_tree(arxiv_id)
    if G is None:
        return None
    type_counts = {}
    has_summaries = 0
    for n in G.nodes:
        t = G.nodes[n].get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
        if G.nodes[n].get('summary', ''):
            has_summaries += 1
    type_counts['nodes_with_summaries'] = has_summaries
    return type_counts


def list_all_papers() -> List[str]:
    """List all arXiv IDs with RAPTOR trees on disk."""
    if not os.path.isdir(TREE_DIR):
        return []
    return sorted(
        f.replace('_tree.gpickle', '')
        for f in os.listdir(TREE_DIR)
        if f.endswith('_tree.gpickle')
    )
