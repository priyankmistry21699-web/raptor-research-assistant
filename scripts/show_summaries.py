"""Show generated summaries for a paper."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.core.raptor_index import get_tree_structure

arxiv_id = sys.argv[1] if len(sys.argv) > 1 else '1706.03762'
struct = get_tree_structure(arxiv_id)
if not struct:
    print(f"No tree found for {arxiv_id}")
    sys.exit(1)

print(f"Paper: {struct.get('title', arxiv_id)}")
print(f"Topics: {struct['total_topics']}, Sections: {struct['total_sections']}, Chunks: {struct['total_chunks']}")
print("=" * 80)

for topic in struct.get('topics', []):
    print(f"\nTOPIC: {topic['title']}")
    summ = topic.get('summary', '')
    print(f"  Summary: {summ[:200]}{'...' if len(summ) > 200 else ''}")
    for sec in topic.get('sections', []):
        print(f"  SECTION: {sec['title']}")
        s = sec.get('summary', '')
        print(f"    Summary: {s[:200]}{'...' if len(s) > 200 else ''}")
