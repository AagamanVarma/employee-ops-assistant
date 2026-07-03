#!/usr/bin/env python3
"""
Verification script for semantic chunking quality.

Inspects chunks in database to validate that semantic chunking is working properly.
"""
import sys
from pathlib import Path
from statistics import mean, stdev

def main():
    """Verify semantic chunking quality."""
    print("\n" + "="*70)
    print("SEMANTIC CHUNKING QUALITY VERIFICATION")
    print("="*70)
    
    # Import app modules
    sys.path.insert(0, str(Path(__file__).parent))
    from app.database import SessionLocal
    from app import models
    
    db = SessionLocal()
    
    # Get all chunks
    chunks = db.query(models.DocumentChunk).all()
    if not chunks:
        print("\n❌ No chunks found in database!")
        print("   → Upload a document first")
        db.close()
        return
    
    print(f"\n✓ Found {len(chunks)} total chunks")
    
    # Analyze chunk statistics
    chunk_words = {}
    chunk_sizes = []
    section_stats = {}
    tiny_chunks = []
    huge_chunks = []
    
    for chunk in chunks:
        # Count words
        word_count = len(chunk.chunk_text.split())
        chunk_sizes.append(word_count)
        
        # Track by document
        if chunk.document_id not in chunk_words:
            chunk_words[chunk.document_id] = []
        chunk_words[chunk.document_id].append(word_count)
        
        # Track by section
        section = chunk.section_title or "Unknown"
        if section not in section_stats:
            section_stats[section] = []
        section_stats[section].append(word_count)
        
        # Flag problems
        if word_count < 50:
            tiny_chunks.append((chunk.document_id, chunk.section_title, word_count, chunk.chunk_text[:50]))
        if word_count > 800:
            huge_chunks.append((chunk.document_id, chunk.section_title, word_count))
    
    # Print statistics
    print("\n" + "-"*70)
    print("OVERALL CHUNK SIZE STATISTICS")
    print("-"*70)
    
    avg_size = mean(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    if len(chunk_sizes) > 1:
        std_dev = stdev(chunk_sizes)
    else:
        std_dev = 0
    
    print(f"  Average chunk size: {avg_size:.0f} words")
    print(f"  Min chunk size: {min_size} words")
    print(f"  Max chunk size: {max_size} words")
    print(f"  Std deviation: {std_dev:.1f}")
    
    # Ideal range check
    print("\n" + "-"*70)
    print("CHUNK SIZE RANGE ANALYSIS")
    print("-"*70)
    
    ideal_min = 100
    ideal_max = 600
    in_range = sum(1 for s in chunk_sizes if ideal_min <= s <= ideal_max)
    too_small = sum(1 for s in chunk_sizes if s < ideal_min)
    too_large = sum(1 for s in chunk_sizes if s > ideal_max)
    
    in_range_pct = (in_range / len(chunks)) * 100
    print(f"  ✓ In ideal range ({ideal_min}-{ideal_max} words): {in_range}/{len(chunks)} ({in_range_pct:.1f}%)")
    print(f"  ⚠️  Too small (< {ideal_min} words): {too_small}")
    print(f"  ⚠️  Too large (> {ideal_max} words): {too_large}")
    
    # Per-document breakdown
    print("\n" + "-"*70)
    print("CHUNKS BY DOCUMENT")
    print("-"*70)
    
    for doc_id, sizes in sorted(chunk_words.items()):
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        doc_name = doc.filename if doc else f"Document {doc_id}"
        avg = mean(sizes)
        print(f"  {doc_name}: {len(sizes)} chunks | Avg: {avg:.0f} words")
    
    # Per-section breakdown
    print("\n" + "-"*70)
    print("CHUNKS BY SECTION")
    print("-"*70)
    
    for section, sizes in sorted(section_stats.items()):
        avg = mean(sizes)
        print(f"  '{section[:40]}': {len(sizes)} chunks | Avg: {avg:.0f} words")
    
    # Problem detection
    print("\n" + "-"*70)
    print("QUALITY CHECKS")
    print("-"*70)
    
    quality_ok = True
    
    # Check 1: Too many tiny chunks
    if too_small > len(chunks) * 0.1:  # More than 10% tiny
        print(f"  ❌ {too_small} chunks are too small (< {ideal_min} words)")
        print("     This indicates poor semantic grouping")
        quality_ok = False
    else:
        print(f"  ✅ Tiny chunks under control ({too_small} found)")
    
    # Check 2: Too many huge chunks
    if too_large > 0:
        print(f"  ⚠️  {too_large} chunks exceed max size ({ideal_max} words)")
        print("     Consider lowering max_words if too many")
        quality_ok = False
    else:
        print(f"  ✅ All chunks within max size limit")
    
    # Check 3: Good distribution
    if in_range_pct > 80:
        print(f"  ✅ {in_range_pct:.0f}% of chunks in ideal range")
    else:
        print(f"  ⚠️  Only {in_range_pct:.0f}% in ideal range (target: 80%+)")
        quality_ok = False
    
    # Check 4: Look for junk chunks
    junk_patterns = ["page", "date", "signature", "approved by", "by:", "©", "®"]
    junk_count = 0
    for chunk in chunks:
        text_lower = chunk.chunk_text.lower()
        if any(pattern in text_lower for pattern in junk_patterns) and len(chunk.chunk_text) < 50:
            junk_count += 1
    
    if junk_count > 0:
        print(f"  ⚠️  {junk_count} potential junk chunks detected")
        quality_ok = False
    else:
        print(f"  ✅ No obvious junk chunks")
    
    # Detailed problem list
    if tiny_chunks:
        print("\n" + "-"*70)
        print("TINY CHUNKS (< 50 words) - These should be rare:")
        print("-"*70)
        for doc_id, section, size, preview in tiny_chunks[:10]:  # Show first 10
            print(f"  [{size:3d} words] Section '{section}': '{preview}...'")
        if len(tiny_chunks) > 10:
            print(f"  ... and {len(tiny_chunks) - 10} more")
    
    if huge_chunks:
        print("\n" + "-"*70)
        print(f"HUGE CHUNKS (> 800 words) - Consider adjusting max_words:")
        print("-"*70)
        for doc_id, section, size in huge_chunks:
            print(f"  [{size} words] Section '{section}'")
    
    # Final verdict
    print("\n" + "="*70)
    if quality_ok and in_range_pct > 80:
        print("✅ SEMANTIC CHUNKING QUALITY: GOOD")
        print("   Chunks are well-sized and semantically grouped")
    elif in_range_pct > 60:
        print("⚠️  SEMANTIC CHUNKING QUALITY: ACCEPTABLE")
        print("   Some tuning recommended (see warnings above)")
    else:
        print("❌ SEMANTIC CHUNKING QUALITY: NEEDS WORK")
        print("   Check issues above and consider adjusting min/max chunk sizes")
    print("="*70)
    
    print("\nNote: If just uploaded, wait a few seconds for indexing to complete.")
    print("      Then re-run this script for accurate results.\n")
    
    db.close()

if __name__ == "__main__":
    main()
