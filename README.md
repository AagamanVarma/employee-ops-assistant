# Employee Ops Assistant

This project is an internal Employee Operations Assistant for policies and workflows.

Current state:
- FastAPI backend
- SQLite storage
- Admin login
- Workflow CRUD
- PDF document upload
- Visual upload progress UI
- Background document processing
- Local document chunking and retrieval
- Local retrieval/debug tooling
- Ask page for policy/workflow search

Important:
- No Gemini/OpenAI response generation yet
- Retrieval is local and must stay gated until confidence is strong enough
- Only highly relevant chunks/workflows should be shown
- If the top score is not strong enough, the app should tell the user to contact HR for clarification
- Policy and workflow matches show references

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

For the best future semantic retrieval model, use a CPU-friendly sentence-transformers model such as `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`.

# Gemini API
Set the environment variable `GEN_API_KEY` to your Google Cloud API key before running the app.

Create a `.env` file in the project root (or set env vars in your shell) and set GEN_API_KEY.

Create the database tables:

```bash
python -m app.init_db
```

Run the server locally:

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir templates --host 127.0.0.1 --port 8000
```

Then open:

http://127.0.0.1:8000/

Demo flow:
- Upload policy PDFs from Admin
- Wait for processing to finish
- Ask a policy question
- If the match is not confident, the app will say to contact HR

Next steps:
1. Add a strict retrieval confidence gate so only strong matches are shown.
2. Tune chunking and ranking further to keep only highly relevant chunks/workflows.
3. Add grounded Gemini/OpenAI summarization after retrieval is stable.
