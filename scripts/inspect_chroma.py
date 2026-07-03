"""
Inspect Chroma vector database: lists collections, shows metadata and sample vectors.
Run: python scripts/inspect_chroma.py
"""
import chromadb
from pathlib import Path
import json

CHROMA_PATH = Path("data/chroma")
if not CHROMA_PATH.exists():
    print(f"Chroma database not found at {CHROMA_PATH}")
    print("Have you uploaded any documents yet?")
    raise SystemExit(1)

try:
    # Connect to Chroma persistent database
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
except Exception as e:
    print(f"Failed to connect to Chroma database: {e}")
    raise SystemExit(1)

# List collections
collections = client.list_collections()
if not collections:
    print("No collections found in Chroma database.")
    raise SystemExit(0)

print(f"\nCollections ({len(collections)}):")
for c in collections:
    print(f"  - {c.name}")

# Inspect each collection
for collection in collections:
    print(f"\n{'='*70}")
    print(f"Collection: {collection.name}")
    print('='*70)
    
    # Collection stats
    count = collection.count()
    print(f"\nStats:")
    print(f"  Total vectors: {count}")
    
    # Peek at metadata
    if count > 0:
        print(f"\nSample vectors (up to 5):")
        
        # Get all IDs and metadata
        try:
            result = collection.get(limit=5, include=["embeddings", "documents", "metadatas"])
            
            ids = result.get("ids", [])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            embeddings = result.get("embeddings", [])
            
            for i, (vid, doc, meta, emb) in enumerate(zip(ids, documents, metadatas, embeddings), 1):
                print(f"\n  [{i}] Vector ID: {vid}")
                print(f"      Document preview: {doc[:80] if doc else '(empty)'}{'...' if doc and len(doc) > 80 else ''}")
                
                if meta:
                    print(f"      Metadata:")
                    for key, value in meta.items():
                        print(f"        - {key}: {value}")
                
                if emb:
                    # Show embedding stats, not the full vector
                    print(f"      Embedding: {len(emb)} dimensions, sample: [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")
        
        except Exception as e:
            print(f"  (error reading vectors): {e}")
    
    # Show metadata schema (infer from first item)
    if count > 0:
        try:
            sample = collection.get(limit=1, include=["metadatas"])
            if sample["metadatas"]:
                meta = sample["metadatas"][0]
                print(f"\nMetadata schema:")
                for key, value in meta.items():
                    print(f"  - {key}: {type(value).__name__}")
        except Exception as e:
            print(f"  (error reading metadata schema): {e}")

print(f"\n{'='*70}")
print("Done.")
print("="*70)
