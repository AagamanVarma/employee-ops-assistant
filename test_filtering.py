#!/usr/bin/env python3
"""Test the post-retrieval filtering layer on problematic queries."""

from app.services.retrieval import retrieve


def test_query(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    
    result = retrieve(query, top_k=3)
    
    intent = result["debug"]["intent_classification"]["intent"]
    confidence = result["debug"]["intent_classification"]["confidence"]
    print(f"Intent: {intent} (confidence: {confidence:.2f})")
    
    chunks = result["chunks"]
    workflows = result["workflows"]
    rejected_chunks = result["debug"].get("rejected_chunks", [])
    rejected_workflows = result["debug"].get("rejected_workflows", [])
    
    print(f"\nAccepted Context:")
    print(f"  - Chunks: {len(chunks)}")
    if chunks:
        for i, chunk in enumerate(chunks, 1):
            print(f"    {i}. {chunk.get('source')} | {chunk.get('section_title')} | Score: {chunk.get('score'):.3f}")
            print(f"       Semantic: {chunk.get('semantic_score'):.3f} | Lexical: {chunk.get('lexical_score'):.3f}")
            print(f"       Entity overlap: {chunk.get('entity_overlap_score', 'N/A')}")
            print(f"       Content: {chunk.get('content', '')[:100]}...")
    
    print(f"  - Workflows: {len(workflows)}")
    if workflows:
        for i, wf in enumerate(workflows, 1):
            print(f"    {i}. {wf.get('title')} | Score: {wf.get('score'):.3f}")
    
    print(f"\nRejected Context (Total: {len(rejected_chunks)} chunks, {len(rejected_workflows)} workflows):")
    if rejected_chunks:
        print(f"  Rejected Chunks ({len(rejected_chunks)}):")
        for i, chunk in enumerate(rejected_chunks[:3], 1):
            print(f"    {i}. {chunk.get('source')}: {chunk.get('rejection_reason')}")
            print(f"       Score: {chunk.get('score'):.3f} | Semantic: {chunk.get('semantic_score'):.3f} | Lexical: {chunk.get('lexical_score'):.3f}")
    
    if rejected_workflows:
        print(f"  Rejected Workflows ({len(rejected_workflows)}):")
        for i, wf in enumerate(rejected_workflows[:3], 1):
            print(f"    {i}. {wf.get('title')}: {wf.get('rejection_reason')}")


if __name__ == "__main__":
    # Test problematic queries
    queries = [
        "what is cricket policy?",
        "what is WFH policy?",
        "How many sick leaves can interns take?",
        "Tell me about FDE role",
        "How do I regularize attendance?",
    ]
    
    for query in queries:
        try:
            test_query(query)
        except Exception as e:
            print(f"ERROR on query '{query}': {e}")
            import traceback
            traceback.print_exc()
