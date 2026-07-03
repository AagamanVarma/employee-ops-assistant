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

### New Metadata Functions

Helper functions detect semantic meaning:

```python
_is_table_like(text)      # Detects tables/structured data
_is_list_item(text)       # Detects bullet points, numbered lists
_is_policy_clause(text)   # Detects numbered clauses (e.g., "4.1 Policy")
```

These preserve chunks that would otherwise be merged.

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

---

## 3. RESETTING OLD DATA (CRITICAL)

The old chunks are now **invalid** and must be cleared completely.

### Step 1: Delete Vector Database

```bash
# The Chroma database stores embeddings
rm -r data/chroma

# Windows PowerShell equivalent:
Remove-Item -Recurse -Force data/chroma
```

**Why**: Old embeddings won't work with new chunks. Starting fresh ensures clean state.

### Step 2: Delete SQLite Database Chunks Table

Run this Python script to clean up the database:

```python
from app.database import SessionLocal, engine
from app import models

# Option A: Clear just document chunks (recommended)
db = SessionLocal()
db.query(models.DocumentChunk).delete()
db.commit()
db.close()

# Option B: Completely reset database (if you want a fresh start)
# models.Base.metadata.drop_all(engine)
# models.Base.metadata.create_all(engine)
```

**Or directly in SQLite**:
```bash
sqlite3 data/chroma/chroma.sqlite3 "DELETE FROM document_chunks;"
```

### Step 3: Verify Clean State

```bash
# Check Chroma vector count (should be 0)
sqlite3 data/chroma/chroma.sqlite3 "SELECT COUNT(*) as vector_count FROM embeddings;"

# Check DocumentChunk table (should be 0)
sqlite3 data/chroma/chroma.sqlite3 "SELECT COUNT(*) as chunk_count FROM document_chunks;"
```

---

## 4. EXACT FILES/FOLDERS TO DELETE

### Delete These:

1. **Vector Store** (Chroma):
   ```
   data/chroma/                    ← Entire directory
   ```
   Contains all old embeddings.

2. **Embedded Documents** (Optional, recommended):
   ```
   data/documents/                 ← If you saved PDFs here
   ```
   Not strictly necessary, but clean slate is better.

### Keep These:

- ✅ `app/services/document_processor.py` (already updated with new chunking)
- ✅ `app/database.py` (unchanged)
- ✅ `app/models.py` (unchanged)
- ✅ `data/chroma/chroma.sqlite3` (will be recreated on upload)

---

## 5. HOW TO RE-UPLOAD DOCUMENTS CLEANLY

### Step 1: Reset Everything

```bash
# PowerShell commands

# Delete old vectors
Remove-Item -Recurse -Force data/chroma

# Delete old document files (optional)
Remove-Item -Recurse -Force data/documents

# Clear database chunks
python -c "
from app.database import SessionLocal
from app import models
db = SessionLocal()
db.query(models.DocumentChunk).delete()
db.commit()
db.close()
print('✓ Database cleaned')
"
```

### Step 2: Restart Application

```bash
# Make sure app is stopped, then restart
cd "path/to/ai-employee-ops-assistant"
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### Step 3: Upload Fresh Documents

1. Go to Admin → Upload Document
2. Select your policy PDF
3. **Watch console logs** for the new chunking output:

```
✓ Semantic chunking for document 1 (Leave_Policy.pdf):
  Section 'Leave Entitlements': 3 chunks | Avg: 180 words | Range: 150-200
  Section 'Extended Leave': 2 chunks | Avg: 170 words | Range: 160-180
  Section 'Leave Request Process': 4 chunks | Avg: 190 words | Range: 175-210
✓ Created 9 semantic chunks (total) for document 1
✓ Indexed 9 semantic chunk vectors in Chroma for document 1
```

This confirms:
- ✅ Semantic chunking working
- ✅ Chunks are in good size range (150-210 words)
- ✅ Vectors indexed successfully

---

## 6. MANUAL TEST QUERIES (AFTER RE-UPLOAD)

### Test Suite

Run these queries to validate chunking quality improvements:

#### Test 1: Specific Policy Detail (Was failing before)

**Query**: "What are leave entitlements?"

**Expected**: Full leave entitlement section with context
**Before**: Scattered tiny fragments
**After**: Complete semantic chunk with dates, amounts, conditions together

---

#### Test 2: Numbered Clause Retrieval (Was breaking before)

**Query**: "What is section 4.1 about?"

**Expected**: Complete "4.1" clause as coherent unit
**Before**: "4.1" alone, then "about" separately, then content fragmented
**After**: "4.1 [Title]" + full policy statement together

---

#### Test 3: Table/Structured Data (Was broken before)

**Query**: "Allowance per shift"

**Expected**: Complete table row with all values
**Before**: Individual cells like "Allowance", "Per Shift", "₹500" as separate chunks
**After**: Entire row together: "Allowance per shift: ₹500 (weekday), ₹750 (night)"

---

#### Test 4: Multi-clause Query (Was noisy before)

**Query**: "Leave policy for interns and WFH rules"

**Expected**: 
- Interns leave clause (150-200 words) 
- WFH policy clause (200-250 words)
- Both with full context

**Before**: 10+ tiny noisy fragments
**After**: 2-3 focused, complete chunks

---

#### Test 5: Edge Case - Very Long Policy

**Query**: "Workflow approval steps"

**Expected**: Workflow steps grouped logically in 2-3 chunks
**Before**: 15+ tiny "step 1", "step 2", "approval by", fragments
**After**: "Step 1: Submit request with [details]", "Step 2: Manager review..."

---

### How to Run Tests

1. Use the Web UI (ask.html):
   - Type query
   - Check results in "Results" tab
   - Look at retrieved chunks in debug panel

2. Check console logs for retrieval behavior:
   ```bash
   # Watch for successful matches
   DEBUG: Retrieved N chunks with avg similarity 0.75+
   ```

3. Verify no orphan chunks appear:
   - Look for "26th Mar 2026" ❌
   - Look for single word fragments ❌
   - Look for complete sentences ✅

---

## 7. DEBUGGING: Monitoring Chunk Quality

### Enable Debug Logging

In `app/main.py` or `.env`:

```python
import logging
logging.getLogger("app.services.document_processor").setLevel(logging.DEBUG)
```

### What to Look For

**Good logs**:
```
✓ Semantic chunking for document 1:
  Section 'Policy': 5 chunks | Avg: 180 words | Range: 150-210
✓ Created 5 semantic chunks (total)
```

**Bad logs** (indicates problem):
```
❌ Section has 0 chunks (empty section)
❌ Chunk size: 8 words (too small, minimum is 100)
```

### Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Too many chunks | Sections too small | Increase max_words to 700 |
| Chunks still tiny | Old code cached | Hard restart, check import |
| Tables broken | Not detected | Check `_is_table_like` logic |
| Policy clauses separated | Regex failed | Debug `_is_policy_clause` |

---

## 8. CONFIGURATION

### Adjust Chunk Sizes

In `document_processor.py`, line ~190:

```python
semantic_chunks = _semantic_chunk_section(
    section["text"], 
    min_words=100,    # ← Adjust minimum
    max_words=600     # ← Adjust maximum
)
```

**Recommendations**:
- `min_words=100`: Good default (eliminates fragments)
- `max_words=600`: Good default (balanced chunks)
- For **large documents**: Try `max_words=800`
- For **detailed policies**: Try `min_words=150`

---

## 9. SUMMARY: WHAT WAS CHANGED

### Files Modified

1. **`app/services/document_processor.py`** ← ONLY FILE CHANGED
   - Old functions replaced:
     - ❌ `_chunk_words()` → removed
     - ❌ `_split_into_sentences()` → removed
   - New functions added:
     - ✅ `_is_table_like()`, `_is_list_item()`, `_is_policy_clause()`
     - ✅ `_count_words()`, `_semantic_chunk_section()`
   - `process_document()` updated to use semantic chunking + debug logging
   - Section extraction logic unchanged (preserved)

### What Stayed the Same

- ✅ Retrieval logic (`retrieval.py`)
- ✅ Reranking logic
- ✅ Query classification
- ✅ Gemini integration
- ✅ Deletion logic
- ✅ Workflow routing
- ✅ Database models
- ✅ Embeddings service

### Data Migration Required

- ❌ Delete `data/chroma/` (vector database)
- ❌ Clear `DocumentChunk` table
- ✅ Re-upload all documents (will use new semantic chunking)

---

## 10. ROLLBACK (If Needed)

If you need to revert to old chunking:

```bash
# Restore from git
git checkout HEAD~1 app/services/document_processor.py

# Or manually restore old code
# (see git history for _chunk_words implementation)
```

However, **this is not recommended**. The old chunking was the source of poor performance. Push through testing instead.

---

## 11. VALIDATION CHECKLIST

After implementation:

- [ ] `document_processor.py` syntax validates
- [ ] `data/chroma/` directory deleted
- [ ] DocumentChunk table cleared
- [ ] Fresh document uploaded successfully
- [ ] Debug logs show proper chunk sizes (100-600 words)
- [ ] Test Query 1 returns complete policies
- [ ] Test Query 2 returns numbered clauses intact
- [ ] Test Query 3 returns table rows whole
- [ ] Test Query 4 returns focused, low-noise results
- [ ] No orphan "26th Mar 2026" type chunks in results
- [ ] Response quality noticeably improved

---

## Questions?

If you encounter issues:

1. Check console logs for chunking stats
2. Verify `data/chroma` was completely deleted
3. Run database cleanup script
4. Hard restart application
5. Re-upload test document
6. Check debug logs match expected output

Good luck! 🚀
