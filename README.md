# Syllabus & Exam Assistant — Scaffold

ConnectX Final Project #5: Grounded RAG + GPA/study-plan calculators + multi-turn memory.

## How it maps to the project brief

| Brief requirement | Where it lives |
|---|---|
| Grounded Syllabus RAG | `backend/rag.py` (FAISS + HuggingFace embeddings) + `search_syllabus` tool in `backend/tools.py` + guardrail prompt in `backend/agent.py` |
| `gpa_impact_simulator` | `backend/tools.py` |
| `generate_study_schedule` | `backend/tools.py` |
| Multi-Turn Context Memory | `SyllabusAssistantAgent._sessions` in `backend/agent.py` |

## Architecture

```
frontend (vanilla JS)  --fetch-->  FastAPI (backend/main.py)
                                        |
                                SyllabusAssistantAgent (backend/agent.py)
                                  - one Groq LLM, bound to 3 tools
                                  - per-session chat history
                                        |
                    -----------------------------------------------
                    |                    |                        |
             search_syllabus     gpa_impact_simulator    generate_study_schedule
             (FAISS retriever)      (pure math)              (pure math)
```

This follows the same **tool-calling ReAct loop** you built in Session 3
(`.bind_tools()` + a loop reading `ai_msg.tool_calls`), just wrapped in a
class that tracks history *per session_id* instead of one notebook-global
list, since a real backend serves more than one student at a time. The
grounding guardrail is the same idea as Assignment 2's
`build_grounded_rag_chain` — "answer only from retrieved context, say so
explicitly if it's missing" — except here the retrieval happens *inside a
tool* the LLM calls, rather than always running before every turn. That
lets the same agent handle GPA and study-plan questions without doing an
unnecessary vector search first.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r req.txt
   ```

2. Export your Groq API key (get one free at console.groq.com/keys):
   ```bash
   export GROQ_API_KEY="gsk_..."          # Windows: set GROQ_API_KEY=gsk_...
   ```

3. Run the backend from the project root:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   First run downloads the `all-MiniLM-L6-v2` embedding model, same as
   Assignment 2 — expect a short pause.

4. Open `frontend/index.html` directly in a browser (or serve it with
   `python -m http.server 5500` from the `frontend/` folder). It talks to
   `http://127.0.0.1:8000` by default — change `API_BASE` at the top of
   `app.js` if you run the backend elsewhere.

## Try it

- "What's the grading breakdown for CS301?"
- "When is the MATH210 final exam?"
- "My current GPA is 3.4 over 60 credits. If I get a B+ in a 4-credit
  class, what's my new GPA?"
- "Build me a 5-day study plan, 3 hours a day, for the CS301 final."
  (the agent should look up CS301's topics first, then call
  `generate_study_schedule`)
- Ask something not in any syllabus (e.g. "what's the campus wifi
  password?") — it should defer to the TA instead of guessing.

## What's stubbed vs. real for a final submission

- **Syllabi are hardcoded** in `backend/data.py` (3 sample courses) — the
  brief implies real syllabus ingestion. A natural next step is a
  `/courses/upload` endpoint that reads a PDF/docx and calls
  `RecursiveCharacterTextSplitter` + `vectorstore.add_documents(...)`.
- **Sessions are in-memory** (`self._sessions` dict in `agent.py`) — fine
  for a demo, but resets whenever the server restarts. Swapping in Redis
  or a small SQLite table would make it persistent.
- **No auth** — `session_id` is just a random UUID generated per browser
  tab in `app.js`.

## File map

```
syllabus_exam_assistant/
├── req.txt
├── README.md
├── backend/
│   ├── __init__.py
│   ├── data.py       # sample syllabus Documents (CS301, MATH210, PSY101)
│   ├── rag.py         # splitter + embeddings + FAISS vectorstore
│   ├── tools.py       # search_syllabus, gpa_impact_simulator, generate_study_schedule
│   ├── agent.py        # SyllabusAssistantAgent (tool-calling loop + memory)
│   ├── models.py       # Pydantic request/response schemas
│   └── main.py         # FastAPI app + endpoints
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```
## How to start
