# Employee Ops Assistant

An internal Employee Operations Assistant for finding policies and workflows quickly.

## Features

- **Admin Panel** - Manage policies and workflows securely with login
- **Policy Upload** - Upload PDF policy documents with visual progress tracking
- **Semantic Chunking** - Documents are split into meaningful semantic chunks (not random fragments) for better retrieval
- **Smart Search** - Ask questions and get relevant policies and workflows instantly
- **Hybrid Retrieval** - Combines both keyword matching and semantic similarity for accurate results
- **Workflow Management** - Create, view, and organize employee workflows
- **Query Classification** - Automatically identifies what type of question you're asking


## Getting Started

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Mac
.\.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

**Note:** On Windows, if sentence-transformers fails, run:
```bash
python -m pip install "numpy<2" "huggingface-hub==0.16.4"
```

### Initialize Database

```bash
python -m app.init_db
```

### Run the Server

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir templates --host 127.0.0.1 --port 8000
```

Then open: **http://127.0.0.1:8000/**

## Using the App

### Upload Policies

1. Go to **Admin → Upload Document**
2. Select a PDF policy file
3. Wait for processing to complete (you'll see progress updates)
4. The policy is now searchable

### Search Policies and Workflows

1. Go to **Ask**
2. Type your question (e.g., "What is the leave policy?" or "How do I request WFH?")
3. View results with relevant excerpts and workflow steps
4. Click on any result for more details

### Manage Workflows

1. Go to **Workflows**
2. Create new workflows or update existing ones
3. Add step-by-step instructions
4. Workflows appear in search results when relevant

## Inspection Tools

### View Database

Inspect SQLite database tables and sample data:

```bash
python scripts/inspect_db.py
```

### View Vector Store

Inspect Chroma vector database and sample embeddings:

```bash
python scripts/inspect_chroma.py
```

## How It Works

1. **Upload** → PDF documents are processed in the background
2. **Chunk** → Documents are split into semantic chunks (complete, meaningful units)
3. **Index** → Chunks are converted to vectors and stored in Chroma
4. **Search** → When you ask a question, the system finds matching chunks using both keywords and semantic similarity
5. **Results** → Top matching policies and workflows are shown with excerpts

## Architecture

- **Backend**: FastAPI
- **Storage**: SQLite (for workflows, admin users, document metadata)
- **Vector DB**: Chroma (for semantic search)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Frontend**: HTML templates with Bootstrap styling
