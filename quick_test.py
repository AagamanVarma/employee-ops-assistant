#!/usr/bin/env python3
"""Quick test of filtering layer - single query."""

from app.services.retrieval import retrieve

queries = [
    'what is cricket policy?',
    'what is WFH policy?',
    'How many sick leaves can interns take?',
    'Tell me about FDE role',
    'How do I regularize attendance?',
]

for query in queries:
    print('\n' + '='*70)
    print('Testing:', repr(query))
    result = retrieve(query, top_k=3)
    print(f"Intent: {result['debug']['intent_classification']['intent']}")
    print(f"Accepted chunks: {len(result['chunks'])}")
    print(f"Accepted workflows: {len(result['workflows'])}")
    print(f"Rejected chunks: {len(result['debug'].get('rejected_chunks', []))}")
    print(f"Rejected workflows: {len(result['debug'].get('rejected_workflows', []))}")
    if result['chunks']:
        chunk = result['chunks'][0]
        print(f"Top chunk: {chunk.get('source')} | Score: {chunk.get('score'):.3f}")
        print(f"  Section: {chunk.get('section_title')}")
        print(f"  Semantic: {chunk.get('semantic_score'):.3f} | Lexical: {chunk.get('lexical_score'):.3f}")
        print(f"  Entity overlap: {chunk.get('entity_overlap_score')}")
    if result['workflows']:
        wf = result['workflows'][0]
        print(f"Top workflow: {wf.get('title')} | Score: {wf.get('score'):.3f}")
    if result['debug'].get('rejected_chunks'):
        rej = result['debug']['rejected_chunks'][0]
        print(f"First rejected chunk reason: {rej.get('rejection_reason')}")
    if result['debug'].get('rejected_workflows'):
        rej = result['debug']['rejected_workflows'][0]
        print(f"First rejected workflow reason: {rej.get('rejection_reason')}")
