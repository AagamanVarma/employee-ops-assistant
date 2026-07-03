#!/usr/bin/env python3
"""
Quick cleanup script for semantic chunking migration.

Safely removes old chunks and vectors to prepare for re-uploading with new semantic chunking.
"""
import os
import shutil
import sys
from pathlib import Path

def confirm(prompt: str) -> bool:
    """Ask for user confirmation."""
    while True:
        response = input(f"\n{prompt} [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no', ''):
            return False
        print("Please enter 'y' or 'n'")

def main():
    """Clean up old chunk data."""
    print("\n" + "="*60)
    print("SEMANTIC CHUNKING MIGRATION CLEANUP")
    print("="*60)
    print("\nThis script will:")
    print("1. Delete Chroma vector database (data/chroma/)")
    print("2. Clear DocumentChunk table from database")
    print("3. Optionally delete old document files")
    print("\n  WARNING: This action cannot be undone!")
    print("  Make sure you have backed up important data!")
    
    # Verify user wants to proceed
    if not confirm("\n Continue with cleanup?"):
        print("\n Cancelled. No changes made.")
        return
    
    # Step 1: Delete Chroma vectors
    chroma_path = Path("data/chroma")
    if chroma_path.exists():
        print(f"\n  Deleting {chroma_path}...")
        try:
            shutil.rmtree(chroma_path)
            print(f" Deleted: {chroma_path}")
        except Exception as e:
            print(f" Failed to delete {chroma_path}: {e}")
            return
    else:
        print(f"\n {chroma_path} not found (already deleted?)")
    
    # Step 2: Clear database chunks
    print("\n  Clearing DocumentChunk table from database...")
    try:
        # Add project to path
        sys.path.insert(0, str(Path(__file__).parent))
        from app.database import SessionLocal
        from app import models
        
        db = SessionLocal()
        count = db.query(models.DocumentChunk).count()
        db.query(models.DocumentChunk).delete()
        db.commit()
        db.close()
        print(f" Deleted {count} chunks from database")
    except Exception as e:
        print(f" Failed to clear DocumentChunk table: {e}")
        return
    
    # Step 3: Optional - delete document files
    docs_path = Path("data/documents")
    if docs_path.exists() and confirm("\n📄 Delete old document files (data/documents/)?"):
        print(f"\n  Deleting {docs_path}...")
        try:
            shutil.rmtree(docs_path)
            docs_path.mkdir(parents=True, exist_ok=True)
            print(f" Deleted: {docs_path}")
        except Exception as e:
            print(f" Failed to delete {docs_path}: {e}")
    
    # Final verification
    print("\n" + "="*60)
    print(" CLEANUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Restart your application")
    print("2. Go to Admin → Upload Document")
    print("3. Re-upload your policy PDFs")
    print("4. Watch console for semantic chunking logs")
    print("5. Test queries to verify improved quality")
    print("\nFor detailed instructions, see: SEMANTIC_CHUNKING_GUIDE.md")

if __name__ == "__main__":
    main()
