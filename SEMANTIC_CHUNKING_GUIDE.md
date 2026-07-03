# Semantic Chunking Implementation Guide

## Overview

The semantic chunking system has been completely redesigned to solve the **poor chunk quality problem**. This document explains the new strategy, how to validate it, and how to reset your data properly.

---

## 1. NEW CHUNKING STRATEGY

### What Changed

**OLD (Broken)**:
- Simple word-count chunking (max 420 words)
- Creates tiny, isolated fragments ("26th Mar 2026", "Approved By")
- Breaks tables and policy clauses apart
- Noisy embeddings from incomplete/meaningless chunks
- No semantic awareness

**NEW (Smart)**:
- **Semantic paragraph grouping**: Groups related paragraphs into meaningful units
- **Minimum chunk size (100 words)**: Eliminates tiny fragments
- **Maximum chunk size (600 words)**: Keeps chunks manageable and focused
- **Intelligent merging**: Small chunks are merged with neighbors unless they're semantically meaningful (tables, lists, policy clauses)
- **Smart edge case handling**:
  - Long paragraphs split by sentence boundaries (not arbitrary word count)
  - Table rows detected and preserved together
  - Numbered policy clauses (e.g., "4.1 Leave Policy") kept intact
  - List items grouped with their context

### Algorithm Overview

```
1. Extract sections by heading (existing logic, preserved)
2. Split section text into paragraphs (double newline = boundary)
3. Group paragraphs into chunks:
   - Add paragraphs until reaching max_words (600)
   - If single paragraph > 600 words, split by sentences
   - Each group becomes a candidate chunk
4. Merge tiny chunks (< 100 words):
   - If chunk is a table/list/policy clause, KEEP IT
   - Otherwise, merge with next chunk to avoid noise
5. Result: Clean, semantically complete chunks
```


---

## 2. WHY THIS IS BETTER

### Problem Solved: Chunk Quality

#### Example: Leave Policy Document

**BEFORE (Bad)**:
```
Chunk 1: "Leave Policy"              [2 words] ← tiny, orphan heading
Chunk 2: "Employees are entitled"    [4 words] ← fragment
Chunk 3: "to 20 days annual"         [4 words] ← fragment
Chunk 4: "26th Mar 2026"             [3 words] ← noisy metadata
Chunk 5: "Approved By: Manager"      [3 words] ← orphan field
Total: 5+ chunks, all weak
```

Result: Weak embeddings, high noise, poor retrieval

**AFTER (Good)**:
```
Chunk 1: "Leave Policy
Employees are entitled to 20 days of annual leave per calendar year.

Interns may take 10 days. This applies to..."
[150 words] ← Semantically complete, good embedding

Chunk 2: "4.1 Extended Leave
Employees on extended leave approved must..."
[180 words] ← Policy clause preserved, meaningful
```

Result: Strong embeddings, low noise, reliable retrieval

### Why Results Improve

1. **Better Embeddings**: Complete sentences and meaningful context → better semantic vectors
2. **Lower Noise**: No tiny fragments → no junk in similarity search
3. **Stable Grounding**: Each chunk has clear context → better for RAG grounding
4. **Policy Integrity**: Clauses stay together → coherent policy statements
5. **Workflow Clarity**: Steps grouped logically → better step-by-step retrieval


## 3. CONFIGURATION

### Adjust Chunk Sizes

In `document_processor.py`, line ~190:

```python
semantic_chunks = _semantic_chunk_section(
    section["text"], 
    min_words=100,    # Adjust minimum
    max_words=600     # Adjust maximum
)
```
---