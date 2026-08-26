"""
FastAPI backend for the Syllabus & Exam Assistant.

Endpoint shapes follow the Demystifying APIs notebook conventions:
Pydantic request/response models, HTTPException for error cases, and a
plain FastAPI() app instance. Run from the project root with:

 To actiate -->  uvicorn backend.main:app --reload --port 8000

v1.1 additions: POST /courses/upload (PDF ingestion) and POST /chat/stream
(SSE streaming chat) -- see their docstrings below for details.
"""
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # connects the front and back ends in browser
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq

from .agent import SyllabusAssistantAgent
from .custom_courses import add_custom_course, add_pdf_course, load_custom_courses, remove_custom_course
from .data import SYLLABUS_DOCUMENTS
from .models import (
    AddCourseRequest,
    ChatRequest,
    ChatResponse,
    CourseInfo,
    ResetResponse,
    SourceChunk,
)
from .pdf_ingest import PDFValidationError, extract_pdf_pages
from .rag import (
    add_course_to_vectorstore,
    add_pdf_course_to_vectorstore,
    build_syllabus_vectorstore,
    remove_course_from_vectorstore,
)
from .tools import make_tools  # imports function that creates the tools used

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
    api_key=GROQ_API_KEY)

# vectorstore starts loaded with the 3 hardcoded syllabi *and* anything a
# student has already saved locally via POST /courses or /courses/upload on
# a previous run.
vectorstore, course_chunk_ids = build_syllabus_vectorstore()
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
    """course_code -> {course_name, custom, source, page_count} for every
    indexed course, base + custom.
    """
    registry: dict[str, dict] = {}
    for doc in SYLLABUS_DOCUMENTS:
        registry[doc.metadata["course_code"]] = {
            "course_name": doc.metadata["course_name"],
            "custom": False,
            "source": "text",
            "page_count": None,
        }
    for course in load_custom_courses():
        pages = course.get("pages")
        registry[course["course_code"]] = {
            "course_name": course["course_name"],
            "custom": True,
            "source": course.get("source", "text"),
            "page_count": len(pages) if pages else None,
        }
    return registry


# Backend security check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "syllabus-exam-assistant",
        "courses_indexed": len(course_chunk_ids),
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
    removed_from_disk = remove_custom_course(code)
    if ids is None and not removed_from_disk:
        raise HTTPException(status_code=404, detail=f"No custom course '{code}' found")

    if ids:
        remove_course_from_vectorstore(vectorstore, ids)
    return {"status": "deleted", "course_code": code}


def _prefixed_input(req: ChatRequest) -> str:
    user_input = req.message
    if req.course_code:
        user_input = f"[Course context: {req.course_code}] {user_input}"
    return user_input


# This is where the frontend sends the users messages
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    result = agent.chat(session_id=req.session_id, user_input=_prefixed_input(req), mode=req.mode)
    return ChatResponse(
        session_id=req.session_id,
        response=result["content"],
        grounded=bool(result["sources"]),
        sources=[SourceChunk(**s) for s in result["sources"]],
        tools_used=result["tools_used"],
        mode=result.get("mode"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
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
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    user_input = _prefixed_input(req)

    def event_stream():
        try:
            for event in agent.chat_stream(session_id=req.session_id, user_input=user_input, mode=req.mode):
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
