# Employee Ops Assistant (Phase 1)

This project is an internal AI-powered Employee Operations Assistant. Phase 1 scaffolds the FastAPI backend, SQLite models, admin authentication, and workflow CRUD with Jinja2 templates.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate  # on mac
.\.venv\Scripts\activate # on Windows
pip install -r requirements.txt
```

Note: on Windows, if sentence-transformers fails, use:

```bash
python -m pip install "numpy<2" "huggingface-hub==0.16.4"
```

# Gemini API
Set the environment variable `GEN_API_KEY` to your Google Cloud API key before running the app.

Create a `.env` file in the project root (or set env vars in your shell). You can copy the example:

```bash
cp .env.example .env
# then edit .env and set GEN_API_KEY
```

Create the database tables:

```bash
python -m app.init_db
```

Create the database tables once before running the app:

```bash
python -m app.init_db
```

Run the server locally:

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir templates --host 127.0.0.1 --port 8000
```

Then open:

http://127.0.0.1:8000/

Next steps: implement RAG ingestion, ChromaDB integration, and AI query pipeline.
