# Employee Ops Assistant

This project is an internal Employee Operations Assistant for policies and workflows.

Current state:
- FastAPI backend
- SQLite storage
- Admin login
- Workflow CRUD
- PDF document upload
- Local document chunking and retrieval
- Ask page for policy/workflow search

Important:
- No Gemini/OpenAI response generation yet
- Retrieval is local and should be tuned before adding generation
- Only confident matches are shown; otherwise the app tells the user to contact HR
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

Demo flow:
- Upload policy PDFs from Admin
- Wait for processing to finish
- Ask a policy question
- If the match is not confident, the app will say to contact HR

Next step after retrieval is strong: add grounded Gemini/OpenAI summarization.
