#!/usr/bin/env python3
"""
Migrate workflow schema to fix reserved keyword issue.

The workflow_steps table previously used "order" (reserved SQL keyword).
Updated schema uses "step_order" instead.

This script safely:
1. Drops old workflow tables (old data is disposable)
2. Preserves all other tables (documents, admins, etc.)
3. Recreates workflow tables with correct schema
"""
import sys
from pathlib import Path
import sqlite3


def main():
    print("\n" + "="*70)
    print("WORKFLOW SCHEMA MIGRATION")
    print("="*70)
    print("\nProblem:")
    print("  SQLite database has old schema: workflow_steps.order")
    print("  SQLAlchemy model expects: workflow_steps.step_order")
    print("\nSolution:")
    print("  Drop old workflow tables and recreate with correct schema")
    print("  (Other tables preserved: documents, admins, etc.)")
    
    # Confirm
    print("\n⚠️  This will DELETE all existing workflows (they are recreatable)")
    response = input("\n✓ Continue? [y/N]: ").strip().lower()
    if response not in ('y', 'yes'):
        print("\n❌ Cancelled. No changes made.")
        return
    
    sys.path.insert(0, str(Path(__file__).parent))
    from app.database import engine, DB_PATH
    from app import models
    
    print(f"\nDatabase: {DB_PATH}")
    
    try:
        # Step 1: Check current schema
        print("\n📋 Checking current schema...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('workflows', 'workflow_steps', 'documents', 'admins')"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        print(f"   Existing tables: {', '.join(sorted(existing_tables))}")
        
        # Check workflow_steps schema
        if 'workflow_steps' in existing_tables:
            cursor.execute("PRAGMA table_info(workflow_steps)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            print(f"   workflow_steps columns: {', '.join(columns.keys())}")
            
            if 'order' in columns:
                print("   ⚠️  Old schema detected (has 'order' column)")
            if 'step_order' in columns:
                print("   ✓ New schema detected (has 'step_order' column)")
        
        conn.close()
        
        # Step 2: Drop old workflow tables
        print("\n🗑️  Dropping old workflow tables...")
        models.WorkflowStep.__table__.drop(engine, checkfirst=True)
        models.Workflow.__table__.drop(engine, checkfirst=True)
        print("   ✅ Dropped workflow_steps")
        print("   ✅ Dropped workflows")
        
        # Step 3: Recreate with correct schema
        print("\n📋 Creating new workflow tables with correct schema...")
        models.Workflow.__table__.create(engine, checkfirst=True)
        models.WorkflowStep.__table__.create(engine, checkfirst=True)
        print("   ✅ Created workflows table")
        print("   ✅ Created workflow_steps table (with step_order column)")
        
        # Step 4: Verify new schema
        print("\n✓ Verifying new schema...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(workflow_steps)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        if 'step_order' in columns and 'order' not in columns:
            print("   ✅ Schema is correct (step_order, no reserved keywords)")
        else:
            print(f"   ⚠️  Unexpected schema: {columns}")
        
        # Verify other tables untouched
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents', 'document_chunks', 'admins')"
        )
        preserved = {row[0] for row in cursor.fetchall()}
        print(f"   ✅ Preserved tables: {', '.join(sorted(preserved))}")
        
        conn.close()
        
        print("\n" + "="*70)
        print("✅ WORKFLOW SCHEMA MIGRATION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Stop the running app (Ctrl+C)")
        print("2. Restart: .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload")
        print("3. Create a new workflow via Admin panel")
        print("4. Verify workflow creation, viewing, and deletion work\n")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("- Ensure app is stopped (stop uvicorn)")
        print("- Ensure database file is not locked")
        print("- Try running the script again")
        return


if __name__ == "__main__":
    main()
