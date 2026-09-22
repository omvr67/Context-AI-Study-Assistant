"""
FastAPI backend for the Syllabus & Exam Assistant.

Endpoint shapes follow the Demystifying APIs notebook conventions:
Pydantic request/response models, HTTPException for error cases, and a
plain FastAPI() app instance. Run from the project root with:

 To actiate -->  uvicorn backend.main:app --reload --port 8000

v1.1 additions: POST /courses/upload (PDF ingestion) and POST /chat/stream
(SSE streaming chat) -- see their docstrings below for details.

v1.3 addition: GET/POST/DELETE /notebook* -- a private, per-device PDF
notebook separate from the shared course library above (see
backend/notebook_store.py). ChatRequest.device_id, when present, binds a
request-scoped search_notebook tool for that one call (see
backend/tools.py's make_notebook_tool) so the agent can search a device's
own notebook without ever being able to search anyone else's.

Solutions addition: POST /courses/{course_code}/solutions and
POST /notebook/{doc_id}/solutions attach an answer-key PDF to an existing
course or notebook doc. Indexed under the same course_code/doc_id as the
source material (so the existing search_syllabus/search_notebook calls
retrieve both together) but tagged material="solutions" so citations --
and the agent's reasoning -- can tell them apart (see rag.format_docs).
course_solutions_chunk_ids / notebook_solutions_chunk_ids track these
chunks separately from each course's/doc's own, so a re-upload can replace
just the solutions instead of duplicating them.

Rate-limiting addition: both /chat and /chat/stream are throttled per
visitor via backend/rate_limit.py (in-memory, ~15 req/min per client
IP) before the agent is ever called, and every Groq call inside the
agent retries with backoff on a 429 -- see backend/agent.py.
"""
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # connects the front and back ends in browser
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq

from .agent import SyllabusAssistantAgent
from .custom_courses import (
    add_course_solutions,
    add_custom_course,
    add_pdf_course,
    get_custom_course,
    load_custom_courses,
    remove_custom_course,
)
from .data import SYLLABUS_DOCUMENTS
from .models import (
    AddCourseRequest,
    ChatRequest,
    ChatResponse,
    CourseInfo,
    NotebookDocInfo,
    ResetResponse,
    SourceChunk,
)
from .notebook_store import (
    add_doc_solutions,
    add_notebook_doc,
    load_all_notebook_docs,
    load_notebook_docs,
    remove_notebook_doc,
)
from .pdf_ingest import PDFValidationError, extract_pdf_pages
from .rag import (
    add_course_solutions_to_vectorstore,
    add_course_to_vectorstore,
    add_notebook_doc_to_vectorstore,
    add_notebook_solutions_to_vectorstore,
    add_pdf_course_to_vectorstore,
    build_syllabus_vectorstore,
    remove_course_from_vectorstore,
)
from .rate_limit import check_and_increment
from .tools import make_notebook_tool, make_tools  # imports functions that create the tools used

# ---------------------------------------------------------------------------
# Startup: build the vector store, tools, and agent once when the app boots.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Export it before starting the server, e.g.\n"
        "  export GROQ_API_KEY='your_key_here'"
    )

llm = ChatGroq(  # creates the LM interface
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=8192, #Increase the max tokens to 8192 for larger context
    api_key=GROQ_API_KEY)

# vectorstore starts loaded with the 3 hardcoded syllabi *and* anything a
# student has already saved locally via POST /courses or /courses/upload on
# a previous run. course_solutions_chunk_ids tracks any attached answer-key
# PDFs separately from each course's own material -- see
# rag.build_syllabus_vectorstore's docstring for why.
vectorstore, course_chunk_ids, course_solutions_chunk_ids = build_syllabus_vectorstore()

# Private per-device notebook: re-index every device's saved uploads (see
# backend/notebook_store.py) into this same live FAISS store, tagged with
# device_id + notebook=True so they can only ever surface through
# search_notebook's own metadata filter -- never through search_syllabus,
# and never for a device other than the one that uploaded them.
# notebook_solutions_chunk_ids is the notebook-side twin of
# course_solutions_chunk_ids above, same reason: tracked separately so a
# re-upload can replace just the solutions chunks for one doc_id.
notebook_chunk_ids: dict[str, list[str]] = {}  # doc_id -> chunk ids
notebook_solutions_chunk_ids: dict[str, list[str]] = {}  # doc_id -> solutions chunk ids
for _device_id, _docs in load_all_notebook_docs().items():
    for _doc in _docs:
        notebook_chunk_ids[_doc["doc_id"]] = add_notebook_doc_to_vectorstore(
            vectorstore, _device_id, _doc["doc_id"], _doc["title"], _doc["pages"]
        )
        _solutions_pages = _doc.get("solutions")
        if _solutions_pages:
            notebook_solutions_chunk_ids[_doc["doc_id"]] = add_notebook_solutions_to_vectorstore(
                vectorstore, _device_id, _doc["doc_id"], _doc["title"], _solutions_pages
            )

tools = make_tools(vectorstore)
agent = SyllabusAssistantAgent(llm=llm, tools=tools)

BASE_COURSE_CODES = {doc.metadata["course_code"] for doc in SYLLABUS_DOCUMENTS}

app = FastAPI(title="ConnectX Syllabus & Exam Assistant")

# Wide-open CORS for local development against the vanilla-JS frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _course_registry() -> dict[str, dict]:
    """course_code -> {course_name, custom, source, page_count, has_solutions}
    for every indexed course, base + custom.
    """
    registry: dict[str, dict] = {}
    for doc in SYLLABUS_DOCUMENTS:
        registry[doc.metadata["course_code"]] = {
            "course_name": doc.metadata["course_name"],
            "custom": False,
            "source": "text",
            "page_count": None,
            "has_solutions": False,
        }
    for course in load_custom_courses():
        pages = course.get("pages")
        registry[course["course_code"]] = {
            "course_name": course["course_name"],
            "custom": True,
            "source": course.get("source", "text"),
            "page_count": len(pages) if pages else None,
            "has_solutions": bool(course.get("solutions")),
        }
    return registry


def _client_key(request: Request) -> str:
    """Best-effort visitor identifier for the rate limiter -- the client IP
    as FastAPI/Starlette sees it. Falls back to a constant if it's ever
    missing (e.g. certain test clients) so throttling degrades to "shared
    across all such requests" instead of crashing.
    """
    return request.client.host if request.client else "unknown"


def _require_device_id(device_id: str | None) -> str:
    """Every /notebook endpoint needs a real device_id -- there's no auth
    system here, so this client-generated id (see frontend/app.js) is the
    entire basis for keeping one device's private uploads away from every
    other device. An empty one is rejected outright rather than silently
    falling back to some shared bucket.
    """
    device_id = (device_id or "").strip()
    if not device_id:
        raise HTTPException(status_code=422, detail="device_id is required")
    return device_id


# Backend security check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "syllabus-exam-assistant",
        "courses_indexed": len(course_chunk_ids),
        "notebook_docs_indexed": len(notebook_chunk_ids),
        "courses_with_solutions": len(course_solutions_chunk_ids),
        "notebook_docs_with_solutions": len(notebook_solutions_chunk_ids),
    }


@app.get("/courses", response_model=list[CourseInfo])
def list_courses():
    registry = _course_registry()
    return [
        CourseInfo(
            course_code=code,
            course_name=info["course_name"],
            custom=info["custom"],
            source=info["source"],
            page_count=info["page_count"],
            has_solutions=info["has_solutions"],
        )
        for code, info in registry.items()
    ]


@app.post("/courses", response_model=CourseInfo, status_code=201)
def add_course(req: AddCourseRequest):
    code = req.course_code.strip().upper()
    name = req.course_name.strip()
    content = req.content.strip()

    if not code or not name or not content:
        raise HTTPException(status_code=422, detail="course_code, course_name, and content are all required")
    if code in BASE_COURSE_CODES:
        raise HTTPException(status_code=409, detail=f"'{code}' is a built-in course and can't be overwritten")
    if len(content) < 30:
        raise HTTPException(status_code=422, detail="content is too short to be a usable syllabus")

    # If this code was already saved before, drop its old chunks first so a
    # re-add fully replaces it instead of leaving stale duplicates indexed.
    if code in course_chunk_ids:
        remove_course_from_vectorstore(vectorstore, course_chunk_ids.pop(code, []))

    new_ids = add_course_to_vectorstore(vectorstore, code, name, content)
    course_chunk_ids[code] = new_ids
    add_custom_course(code, name, content)

    return CourseInfo(course_code=code, course_name=name, custom=True, source="text")


@app.post("/courses/upload", response_model=CourseInfo, status_code=201)
async def upload_course_pdf(
    course_code: str = Form(...),
    course_name: str = Form(...),
    append: bool = Form(False),
    file: UploadFile = File(...),
):
    """Uploads a PDF syllabus/chapter, validates it, extracts page-aware
    text, and adds it to the live FAISS index so it becomes usable by the
    existing RAG (search_syllabus, build_ai_study_plan, etc.) immediately --
    no restart needed.

    append=True adds the new pages after any existing pages already saved
    under this course_code (growing the material, e.g. uploading chapter 2
    after chapter 1). append=False (default) replaces any earlier version
    under this code entirely, matching how the pasted-text POST /courses
    already behaves.
    """
    code = course_code.strip().upper()
    name = course_name.strip()

    if not code or not name:
        raise HTTPException(status_code=422, detail="course_code and course_name are required")
    if code in BASE_COURSE_CODES:
        raise HTTPException(status_code=409, detail=f"'{code}' is a built-in course and can't be overwritten")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await file.read()
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not append and code in course_chunk_ids:
        remove_course_from_vectorstore(vectorstore, course_chunk_ids.pop(code, []))

    new_ids = add_pdf_course_to_vectorstore(vectorstore, code, name, pages)
    if append and code in course_chunk_ids:
        course_chunk_ids[code] = course_chunk_ids[code] + new_ids
    else:
        course_chunk_ids[code] = new_ids

    saved = add_pdf_course(code, name, pages, append=append)
    total_pages = len(saved["pages"])

    return CourseInfo(course_code=code, course_name=name, custom=True, source="pdf", page_count=total_pages)


@app.delete("/courses/{course_code}")
def delete_course(course_code: str):
    code = course_code.strip().upper()
    if code in BASE_COURSE_CODES:
        raise HTTPException(status_code=403, detail="Built-in courses can't be deleted")

    ids = course_chunk_ids.pop(code, None)
    solutions_ids = course_solutions_chunk_ids.pop(code, None)
    removed_from_disk = remove_custom_course(code)
    if ids is None and solutions_ids is None and not removed_from_disk:
        raise HTTPException(status_code=404, detail=f"No custom course '{code}' found")

    if ids:
        remove_course_from_vectorstore(vectorstore, ids)
    if solutions_ids:
        remove_course_from_vectorstore(vectorstore, solutions_ids)
    return {"status": "deleted", "course_code": code}


@app.post("/courses/{course_code}/solutions", response_model=CourseInfo, status_code=201)
async def upload_course_solutions(course_code: str, file: UploadFile = File(...)):
    """Attaches a solutions/answer-key PDF to an existing custom course.
    Indexed under the same course_code as the course's own material, but
    tagged material="solutions" so search_syllabus's citations -- and the
    agent's own reasoning -- can tell the two apart even though a search
    retrieves them together (see rag.format_docs and
    GUARDRAIL_SYSTEM_PROMPT's rule 3). Re-uploading replaces any
    previously-attached solutions rather than stacking duplicates: any
    chunks already tracked under this course_code in
    course_solutions_chunk_ids are removed first.

    Only works on a *custom* (already-added) course -- there's nothing
    meaningful to attach an answer key to on a built-in course's plain
    grading-policy syllabus.
    """
    code = course_code.strip().upper()
    if code in BASE_COURSE_CODES:
        raise HTTPException(
            status_code=409, detail=f"'{code}' is a built-in course and can't take an attached solutions PDF"
        )
    existing = get_custom_course(code)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No custom course '{code}' found -- add the course itself first")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await file.read()
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if code in course_solutions_chunk_ids:
        remove_course_from_vectorstore(vectorstore, course_solutions_chunk_ids.pop(code, []))

    new_ids = add_course_solutions_to_vectorstore(vectorstore, code, existing["course_name"], pages)
    course_solutions_chunk_ids[code] = new_ids
    add_course_solutions(code, pages)

    info = _course_registry()[code]
    return CourseInfo(
        course_code=code,
        course_name=info["course_name"],
        custom=True,
        source=info["source"],
        page_count=info["page_count"],
        has_solutions=True,
    )


# --- Private per-device notebook -------------------------------------------
# Separate from the shared course library above: a device_id generated and
# stored client-side (see frontend/app.js) scopes every one of these
# endpoints, and the same id is threaded through to make_notebook_tool at
# chat time so an uploaded PDF is only ever searchable by the device that
# uploaded it -- see backend/notebook_store.py and backend/tools.py.

@app.get("/notebook", response_model=list[NotebookDocInfo])
def list_notebook(device_id: str = Query(...)):
    device_id = _require_device_id(device_id)
    return [
        NotebookDocInfo(
            doc_id=d["doc_id"],
            title=d["title"],
            page_count=d.get("page_count", len(d.get("pages", []))),
            has_solutions=bool(d.get("solutions")),
        )
        for d in load_notebook_docs(device_id)
    ]


@app.post("/notebook/upload", response_model=NotebookDocInfo, status_code=201)
async def upload_notebook_pdf(
    device_id: str = Form(...),
    title: str = Form(...),
    file: UploadFile = File(...),
):
    """Uploads a private PDF to the caller's own notebook. Indexed into the
    same live FAISS store the shared courses use, but tagged with this
    device_id so it's only ever found by a search_notebook call made on
    this device's behalf -- never by search_syllabus, and never by another
    device's notebook search. No restart needed; it's searchable immediately.
    """
    device_id = _require_device_id(device_id)
    doc_title = title.strip()
    if not doc_title:
        raise HTTPException(status_code=422, detail="title is required")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await file.read()
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    saved = add_notebook_doc(device_id, doc_title, pages)
    notebook_chunk_ids[saved["doc_id"]] = add_notebook_doc_to_vectorstore(
        vectorstore, device_id, saved["doc_id"], doc_title, pages
    )

    return NotebookDocInfo(doc_id=saved["doc_id"], title=doc_title, page_count=saved["page_count"])


@app.delete("/notebook/{doc_id}")
def delete_notebook_doc(doc_id: str, device_id: str = Query(...)):
    device_id = _require_device_id(device_id)

    # Ownership check happens against the store itself, before touching
    # anything else -- a doc_id that exists but belongs to a different
    # device_id is treated identically to one that doesn't exist at all,
    # so this response can never confirm or deny another device's upload.
    owned_ids = {d["doc_id"] for d in load_notebook_docs(device_id)}
    if doc_id not in owned_ids:
        raise HTTPException(status_code=404, detail=f"No notebook document '{doc_id}' found")

    remove_notebook_doc(device_id, doc_id)
    ids = notebook_chunk_ids.pop(doc_id, None)
    solutions_ids = notebook_solutions_chunk_ids.pop(doc_id, None)
    if ids:
        remove_course_from_vectorstore(vectorstore, ids)
    if solutions_ids:
        remove_course_from_vectorstore(vectorstore, solutions_ids)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/notebook/{doc_id}/solutions", response_model=NotebookDocInfo, status_code=201)
async def upload_notebook_solutions(doc_id: str, device_id: str = Form(...), file: UploadFile = File(...)):
    """Notebook equivalent of POST /courses/{course_code}/solutions: attaches
    a solutions/answer-key PDF to one of the caller's own notebook docs.
    Ownership is checked the same way delete_notebook_doc checks it -- a
    doc_id belonging to a different device_id is treated as not found, so
    this can never confirm or attach anything to another device's upload.
    Re-uploading replaces any previously-attached solutions for this doc_id
    rather than stacking duplicates.
    """
    device_id = _require_device_id(device_id)

    owned = {d["doc_id"]: d for d in load_notebook_docs(device_id)}
    existing = owned.get(doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No notebook document '{doc_id}' found")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await file.read()
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if doc_id in notebook_solutions_chunk_ids:
        remove_course_from_vectorstore(vectorstore, notebook_solutions_chunk_ids.pop(doc_id, []))

    new_ids = add_notebook_solutions_to_vectorstore(vectorstore, device_id, doc_id, existing["title"], pages)
    notebook_solutions_chunk_ids[doc_id] = new_ids
    add_doc_solutions(device_id, doc_id, pages)

    return NotebookDocInfo(
        doc_id=doc_id,
        title=existing["title"],
        page_count=existing.get("page_count", len(existing.get("pages", []))),
        has_solutions=True,
    )


def _prefixed_input(req: ChatRequest) -> str:
    user_input = req.message
    if req.course_code:
        user_input = f"[Course context: {req.course_code}] {user_input}"
    return user_input


# This is where the frontend sends the users messages
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    allowed, retry_after = check_and_increment(_client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"You're sending messages a little fast -- please wait {retry_after}s and try again.",
        )

    extra_tools = [make_notebook_tool(vectorstore, req.device_id)] if req.device_id else None
    result = agent.chat(session_id=req.session_id, user_input=_prefixed_input(req), mode=req.mode, extra_tools=extra_tools)
    return ChatResponse(
        session_id=req.session_id,
        response=result["content"],
        grounded=bool(result["sources"]),
        sources=[SourceChunk(**s) for s in result["sources"]],
        tools_used=result["tools_used"],
        mode=result.get("mode"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """Streaming twin of POST /chat, as Server-Sent Events. Each event is a
    line `data: <json>\\n\\n` with one of these shapes:

      {"type": "tool", "name": "search_syllabus"}
      {"type": "token", "text": "..."}
      {"type": "done", "content", "grounded", "sources", "tools_used", "mode"}
      {"type": "error", "error": "..."}

    Tool orchestration is identical to /chat -- the same guardrails, same
    tools, same grounding behavior. Cancellation is handled by the client
    simply aborting the fetch/EventSource; if the connection drops mid-turn,
    the generator raises GeneratorExit on its next yield and this function
    lets that propagate to stop cleanly rather than trying to keep writing
    to a closed connection.

    The rate-limit check happens here, before the StreamingResponse is
    even constructed, so a throttled request comes back as a normal JSON
    429 the frontend already knows how to parse -- not a malformed or
    empty SSE stream.
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    allowed, retry_after = check_and_increment(_client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"You're sending messages a little fast -- please wait {retry_after}s and try again.",
        )

    user_input = _prefixed_input(req)
    extra_tools = [make_notebook_tool(vectorstore, req.device_id)] if req.device_id else None

    def event_stream():
        try:
            for event in agent.chat_stream(session_id=req.session_id, user_input=user_input, mode=req.mode, extra_tools=extra_tools):
                if event.get("type") == "done":
                    event = {**event, "grounded": bool(event.get("sources"))}
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Resets the session, optionally returning a short recap first
@app.delete("/chat/{session_id}", response_model=ResetResponse)
def reset_session(session_id: str, summarize: bool = Query(False)):
    summary = agent.summarize(session_id) if summarize else None
    agent.reset(session_id)
    return ResetResponse(status="reset", session_id=session_id, summary=summary)
