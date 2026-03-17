"""
Tests for RAPTOR Hierarchical Index — all core modules.
Run: python tests/test_raptor_index.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        msg = f"[PASS] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
    else:
        failed += 1
        msg = f"[FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ============================================================
# TEST 1: raptor_index — Core Module
# ============================================================
print("=" * 60)
print("TEST 1: raptor_index — Core Module")
print("=" * 60)

from app.core.raptor_index import (
    load_tree, save_tree, get_paper_info, get_topics, get_sections,
    get_chunks, get_tree_structure, get_context_for_chunk, get_tree_stats,
    list_all_papers
)

# 1a. list all papers
papers = list_all_papers()
test("list_all_papers", len(papers) > 0, f"{len(papers)} papers")

# 1b. load valid tree
G = load_tree('1706.03762')
test("load_tree(valid)", G is not None, f"{len(G.nodes)} nodes")

# 1c. load invalid tree
G_none = load_tree('NONEXISTENT_PAPER')
test("load_tree(invalid)", G_none is None, "returns None")

# 1d. get_paper_info
info = get_paper_info(G)
test("get_paper_info", info['arxiv_id'] == '1706.03762' and 'Attention' in info['title'],
     info['title'][:50])

# 1e. get_topics
topics = get_topics(G)
test("get_topics", len(topics) > 0, f"{len(topics)} topics")
for t in topics:
    print(f"       - {t['title']} ({t['section_count']} sections)")

# 1f. get_sections — all
sections = get_sections(G)
test("get_sections(all)", len(sections) > 0, f"{len(sections)} sections")

# 1g. get_sections — filtered by topic
if topics:
    secs_filtered = get_sections(G, topics[0]['node_id'])
    test("get_sections(topic)", len(secs_filtered) > 0,
         f"{len(secs_filtered)} sections under '{topics[0]['title']}'")

# 1h. get_chunks
if sections:
    chunks = get_chunks(G, sections[0]['node_id'])
    test("get_chunks", len(chunks) > 0 and all('text' in c for c in chunks),
         f"{len(chunks)} chunks under '{sections[0]['title']}'")

# 1i. get_tree_structure — 4-level tree
struct = get_tree_structure('1706.03762')
test("get_tree_structure(4-level)",
     struct is not None and struct['tree_levels'] == 4 and struct['total_topics'] > 0,
     f"levels={struct['tree_levels']}, topics={struct['total_topics']}, "
     f"sections={struct['total_sections']}, chunks={struct['total_chunks']}")

# 1j. get_context_for_chunk
chunk_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'chunk']
ctx = get_context_for_chunk(G, chunk_nodes[50])
test("get_context_for_chunk",
     ctx['paper_title'] != '' and (ctx['topic'] != '' or ctx['section_title'] != ''),
     f"topic='{ctx['topic']}', section='{ctx['section_title']}'")

# 1k. get_tree_stats
stats = get_tree_stats('1706.03762')
test("get_tree_stats", stats is not None and stats.get('topic', 0) > 0,
     str(stats))

# 1l. 3-level tree (paper without topics)
papers_no_topics = [p for p in papers if (get_tree_stats(p) or {}).get('topic', 0) == 0]
if papers_no_topics:
    s3 = get_tree_structure(papers_no_topics[0])
    test("get_tree_structure(3-level)", s3 is not None and s3['tree_levels'] == 3,
         f"{papers_no_topics[0]}: chunks={s3['total_chunks']}, level=3")
else:
    print("  (no 3-level trees to test)")


# ============================================================
# TEST 2: vector_db — VectorDB
# ============================================================
print()
print("=" * 60)
print("TEST 2: vector_db — VectorDB")
print("=" * 60)

try:
    from app.core.vector_db import VectorDB
    db = VectorDB()
    count = db.count()
    test("VectorDB.count()", count >= 0, f"{count} documents")

    if count > 0:
        from app.core.embedding import EmbeddingModel
        embedder = EmbeddingModel()
        qvec = embedder.encode("transformer attention mechanism")

        # search
        results = db.search(qvec, top_k=3)
        test("VectorDB.search()", len(results) > 0,
             f"{len(results)} results, top: {results[0]['arxiv_id']}")
        test("VectorDB.search() fields",
             all(k in results[0] for k in ['id', 'text', 'metadata', 'distance', 'arxiv_id']),
             "all fields present")

        # search_by_paper
        results_paper = db.search_by_paper(qvec, '1706.03762', top_k=3)
        test("VectorDB.search_by_paper()",
             len(results_paper) > 0 and all(r['arxiv_id'] == '1706.03762' for r in results_paper),
             f"{len(results_paper)} results from 1706.03762")

        # get_by_id
        doc = db.get_by_id(results[0]['id'])
        test("VectorDB.get_by_id()", doc is not None and 'text' in doc,
             f"id={doc['id'][:30]}")
    else:
        print("  (Chroma empty — skipping search tests)")

except ImportError as e:
    print(f"  [SKIP] chromadb not installed: {e}")
except Exception as e:
    test("VectorDB init", False, str(e))


# ============================================================
# TEST 3: retrieval — RaptorRetriever
# ============================================================
print()
print("=" * 60)
print("TEST 3: retrieval — RaptorRetriever")
print("=" * 60)

try:
    from app.core.retrieval import RaptorRetriever

    retriever = RaptorRetriever()
    test("RaptorRetriever init", True)

    # list papers
    available = retriever.list_available_papers()
    test("list_available_papers()", len(available) > 0, f"{len(available)} papers")

    # paper overview
    overview = retriever.get_paper_overview('1706.03762')
    test("get_paper_overview()", overview is not None and overview['tree_levels'] == 4,
         f"'{overview['title'][:40]}', {overview['total_topics']} topics")

    # tree-based retrieval by section
    tree_results = retriever.retrieve_by_tree('1706.03762', section='Introduction')
    test("retrieve_by_tree(section)", len(tree_results) > 0,
         f"{len(tree_results)} chunks from Introduction")

    # tree-based retrieval by topic
    tree_results_topic = retriever.retrieve_by_tree('1706.03762', topic='Background')
    test("retrieve_by_tree(topic)", len(tree_results_topic) > 0,
         f"{len(tree_results_topic)} chunks from Background topic")

    # hybrid retrieval (requires Chroma)
    if db.count() > 0:
        results = retriever.retrieve("What is the transformer architecture?", top_k=3)
        test("retrieve(hybrid)", len(results) > 0, f"{len(results)} enriched results")

        if results:
            r = results[0]
            test("retrieve result has tree_context",
                 'tree_context' in r and 'context_text' in r,
                 f"topic='{r['tree_context'].get('topic', '')}', "
                 f"section='{r['tree_context'].get('section_title', '')}'")
            print(f"\n  --- Sample context_text (first 300 chars) ---")
            print(f"  {r['context_text'][:300]}...")

        # paper-specific retrieval
        results_paper = retriever.retrieve(
            "self-attention mechanism", top_k=3, arxiv_id='1706.03762')
        test("retrieve(paper-specific)",
             len(results_paper) > 0,
             f"{len(results_paper)} results from 1706.03762")

        # retrieval without tree context
        results_plain = retriever.retrieve(
            "neural network", top_k=2, include_tree_context=False)
        test("retrieve(no tree context)",
             len(results_plain) > 0 and 'tree_context' not in results_plain[0],
             f"{len(results_plain)} plain results")
    else:
        print("  (Chroma empty — skipping hybrid retrieval tests)")

except ImportError as e:
    print(f"  [SKIP] dependency missing: {e}")
except Exception as e:
    import traceback
    test("RaptorRetriever", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 4: build_index pipeline (dry check — imports only)
# ============================================================
print()
print("=" * 60)
print("TEST 4: build_index — Pipeline Import Check")
print("=" * 60)

try:
    # Only test that the script's functions can be imported
    sys.argv = ['build_index.py', '--step', 'trees', '--paper', 'NONEXISTENT']
    exec(open('scripts/build_index.py').read().split('if __name__')[0])
    test("build_index imports", True, "all functions defined")
except Exception as e:
    test("build_index imports", False, str(e))


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
