"""
Inspect SQLite database: lists tables, shows schema and sample rows.
Run: python scripts/inspect_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/app.db")
if not DB_PATH.exists():
    print(f"Database not found at {DB_PATH}. Did you run: python -m app.init_db ?")
    raise SystemExit(1)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# list tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
rows = cur.fetchall()
if not rows:
    print("No tables found in database.")
    conn.close()
    raise SystemExit(0)

tables = [r[0] for r in rows]
print("Tables:")
for t in tables:
    print(" -", t)

for t in tables:
    print("\n--- Schema for", t, "---")
    cur.execute(f"PRAGMA table_info({t});")
    cols = cur.fetchall()
    for c in cols:
        print(f"  {c['cid']}: {c['name']} {c['type']} (notnull={c['notnull']}, default={c['dflt_value']})")

    print("\nSample rows:")
    try:
        cur.execute(f"SELECT * FROM {t} LIMIT 5;")
        sample = cur.fetchall()
        if not sample:
            print("  (no rows)")
        else:
            for r in sample:
                print("  ", dict(r))
    except Exception as e:
        print("  (error reading rows)", e)

conn.close()
print("\nDone.")
