"""
FastAPI backend for the Syllabus & Exam Assistant.

Endpoint shapes follow the Demystifying APIs notebook conventions:
Pydantic request/response models, HTTPException for error cases, and a
plain FastAPI() app instance. Run from the project root with:

 To actiate -->  uvicorn backend.main:app --reload --port 8000
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware  # connects the front and back ends in browser
from langchain_groq import ChatGroq

from .agent import SyllabusAssistantAgent
from .custom_courses import add_custom_course, load_custom_courses, remove_custom_course
from .data import SYLLABUS_DOCUMENTS
from .models import (
    AddCourseRequest,
    ChatRequest,
    ChatResponse,
    CourseInfo,
    ResetResponse,
    SourceChunk,
)
from .rag import add_course_to_vectorstore, build_syllabus_vectorstore, remove_course_from_vectorstore
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
# student has already saved locally via POST /courses on a previous run.
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
    """course_code -> {course_name, custom} for every indexed course, base + custom."""
    registry: dict[str, dict] = {}
    for doc in SYLLABUS_DOCUMENTS:
        registry[doc.metadata["course_code"]] = {"course_name": doc.metadata["course_name"], "custom": False}
    for course in load_custom_courses():
        registry[course["course_code"]] = {"course_name": course["course_name"], "custom": True}
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
        CourseInfo(course_code=code, course_name=info["course_name"], custom=info["custom"])
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

    return CourseInfo(course_code=code, course_name=name, custom=True)


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


# This is where the frontend sends the users messages
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    user_input = req.message
    if req.course_code:
        user_input = f"[Course context: {req.course_code}] {user_input}"

    result = agent.chat(session_id=req.session_id, user_input=user_input)
    return ChatResponse(
        session_id=req.session_id,
        response=result["content"],
        grounded=bool(result["sources"]),
        sources=[SourceChunk(**s) for s in result["sources"]],
        tools_used=result["tools_used"],
    )


# Resets the session, optionally returning a short recap first
@app.delete("/chat/{session_id}", response_model=ResetResponse)
def reset_session(session_id: str, summarize: bool = Query(False)):
    summary = agent.summarize(session_id) if summarize else None
    agent.reset(session_id)
    return ResetResponse(status="reset", session_id=session_id, summary=summary)
