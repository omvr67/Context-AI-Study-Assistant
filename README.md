# Syllabus & Exam Assistant 

ConnectX Final Project #5: Grounded RAG + GPA/study-plan calculators + multi-turn memory.

## v1.1

Built on top of the same tool-calling agent rather than bolted onto it --
the agent still decides when to invoke a tool instead of retrieving on
every message; v1.1 mostly adds *more* tools and *more* ways to feed the
RAG store, plus a real streaming transport.

- **PDF upload** (`POST /courses/upload`) -- validates the file, extracts
  text page-by-page (`backend/pdf_ingest.py`), chunks + embeds it, and adds
  it to the live FAISS index immediately. Chunks keep their source page
  number, so `search_syllabus` can cite e.g. `CS301 p.4`.
- **Explain Like I'm Stupid / Exam Mode / Teach Me This Chapter** -- three
  explicit interaction modes (`ChatRequest.mode`), each injected as an
  ephemeral instruction for just that turn (`backend/agent.py`'s
  `MODE_PROMPTS`). None of them touch the grounding rules -- a mode changes
  *how* the assistant explains, never what it's allowed to assert.
- **AI Study Planner** (`build_ai_study_plan` tool) -- given just a
  `course_code`, pulls the topic list straight out of the syllabus
  (`get_course_full_text` + a `Week N: Topic` regex) and computes days
  remaining from an exam date, instead of requiring topics to be retyped.
  The original `generate_study_schedule` tool is kept for when a student
  hands over an explicit topic list of their own.
- **GPA Target Planner** (`gpa_target_planner` tool) -- given a goal GPA,
  works backward to the average grade needed across remaining credits,
  alongside the existing `gpa_impact_simulator` (which only simulates one
  already-decided grade).
- **Better RAG** -- `rag.retrieve_relevant_chunks()` adds a minimum
  relevance floor (so `search_syllabus` can actually say "not found"
  instead of always returning its k-nearest chunks regardless of quality)
  and a per-course cap on results (so one heavily-indexed course can't
  crowd out every slot in an all-courses search). Page-aware metadata
  flows through PDF-sourced chunks end to end.
- **Streaming** (`POST /chat/stream`, Server-Sent Events) -- tool-call
  orchestration is unchanged; the model's final tool-free answer streams
  token-by-token to the frontend, which renders it incrementally and
  supports mid-stream cancellation (a "Stop" button that aborts the fetch).
  `POST /chat` (non-streaming) is still there and unchanged, for anything
  that doesn't want an SSE client.
- **Teach Me This Chapter** is one of the three modes above -- it forces a
  fixed explanation → example → understanding-check → practice structure,
  grounded in whatever `search_syllabus` retrieves for the requested topic.
- **Autocorrect: left out.** A typo-correction layer that's actually
  comparable to Notes/Gemini-style contextual correction needs either a
  hosted spellcheck model or a large offline dictionary + language model --
  both are the "heavy/fragile dependency" this project was trying to avoid.
  A pure-regex/edit-distance approach would be there mainly to *look*
  implemented while silently mis-correcting course codes, names, and
  jargon it has no way to recognize -- worse than not having it. Left out
  rather than shipped as a worse approximation.

## How it maps to the project brief

| Brief requirement | Where it lives |
|---|---|
| Grounded Syllabus RAG | `backend/rag.py` (FAISS + HuggingFace embeddings) + `search_syllabus` tool in `backend/tools.py` + guardrail prompt in `backend/agent.py` |
| `gpa_impact_simulator` / `gpa_target_planner` | `backend/tools.py` |
| `generate_study_schedule` / `build_ai_study_plan` | `backend/tools.py` |
| Multi-Turn Context Memory | `SyllabusAssistantAgent._sessions` in `backend/agent.py` |
| PDF ingestion | `backend/pdf_ingest.py` + `POST /courses/upload` |
| Streaming | `SyllabusAssistantAgent.chat_stream` + `POST /chat/stream` |

## Architecture

```
frontend (vanilla JS)  --fetch/SSE-->  FastAPI (backend/main.py)
                                        |
                                SyllabusAssistantAgent (backend/agent.py)
                                  - one Groq LLM, bound to 5 tools
                                  - per-session chat history
                                  - optional ephemeral mode (ELI5/Exam/Teach)
                                        |
        -----------------------------------------------------------------
        |              |                  |                |             |
 search_syllabus  gpa_impact_    gpa_target_   generate_study_   build_ai_
 (FAISS + floor)  simulator      planner       schedule          study_plan
                  (pure math)    (pure math)   (pure math)       (regex + math)
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

2. Configure your Groq API key. Get a free key from
   [Groq Console](https://console.groq.com/keys); never commit a real key.
   Copy `.env.example` to `.env`, replace the placeholder with your key, and
   load it into your shell before starting the backend:
   ```bash
   cp .env.example .env
   # Edit .env, then:
   set -a; source .env; set +a
   ```
   Alternatively, set it directly:
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
   `app.js` if you run the backend elsewhere. I would also recommend downloading the 'Live Server' extention, right clicking on the index.html file in the frontend and choosing open with live server.

## Secrets for CI and deployment

Store `GROQ_API_KEY` in your platform's secret manager rather than in source
control. For GitHub Actions, add it as a repository secret and provide it to
the runtime as `${{ secrets.GROQ_API_KEY }}`. Rotate a key promptly if it is
ever exposed.

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

- **The 3 sample syllabi are still hardcoded** in `backend/data.py` — but
  real ingestion now exists for everything added on top of them: pasted
  text via `POST /courses`, and PDFs (page-aware) via
  `POST /courses/upload` (see `backend/pdf_ingest.py`). `.docx` is not
  handled yet — only PDF, per the v1.1 scope.
- **Sessions are in-memory** (`self._sessions` dict in `agent.py`) — fine
  for a demo, but resets whenever the server restarts. Swapping in Redis
  or a small SQLite table would make it persistent.
- **No auth** — `session_id` is just a random UUID generated per browser
  tab in `app.js`.

## Try it (v1.1 additions)

- Upload a lecture-notes PDF under a new course code in the sidebar, then
  ask a question about its content — no restart needed.
- Toggle "Explain Like I'm 5" and ask the same syllabus question you'd
  normally ask — same grounding, simpler explanation.
- Toggle "Exam Mode" and ask "quiz me on CS301" — expect exam-relevant
  framing and a practice question grounded in the real syllabus.
- Toggle "Teach Me This Chapter" and ask about a specific lecture topic —
  expect explanation → example → understanding check → practice, in order.
- "What GPA do I need to reach a 3.7 with 30 credits left?" — routes to
  `gpa_target_planner` instead of `gpa_impact_simulator`.
- "Build me a study plan for CS301's final" with no topics given — routes
  to `build_ai_study_plan`, which finds CS301's topics automatically.
- Send a message and hit **Stop** partway through — the stream aborts
  cleanly and the partial answer stays visible, marked as stopped.

## File map

```
syllabus_exam_assistant/
├── req.txt
├── README.md
├── backend/
│   ├── __init__.py
│   ├── data.py       # sample syllabus Documents (CS301, MATH210, PSY101)
│   ├── rag.py         # splitter + embeddings + FAISS vectorstore + relevance-floor retrieval
│   ├── pdf_ingest.py   # PDF validation + page-aware text extraction (v1.1)
│   ├── custom_courses.py # local JSON persistence for pasted-text + PDF courses
│   ├── tools.py       # search_syllabus, gpa_impact_simulator, gpa_target_planner,
│   │                   # generate_study_schedule, build_ai_study_plan
│   ├── agent.py        # SyllabusAssistantAgent (tool-calling loop + memory + modes + streaming)
│   ├── models.py       # Pydantic request/response schemas
│   └── main.py         # FastAPI app + endpoints (incl. /courses/upload, /chat/stream)
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```
