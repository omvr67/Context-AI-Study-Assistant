"""
FastAPI backend for the Syllabus & Exam Assistant.

Endpoint shapes follow the Demystifying APIs notebook conventions:
Pydantic request/response models, HTTPException for error cases, and a
plain FastAPI() app instance. Run from the project root with:

 To actiate -->  uvicorn backend.main:app --reload --port 8000
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq

from .agent import SyllabusAssistantAgent
from .data import SYLLABUS_DOCUMENTS
from .models import ChatRequest, ChatResponse, CourseInfo
from .rag import build_syllabus_vectorstore
from .tools import make_tools

# ---------------------------------------------------------------------------
# Startup: build the vector store, tools, and agent once when the app boots.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Export it before starting the server, e.g.\n"
        "  export GROQ_API_KEY='your_key_here'"
    )

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, api_key=GROQ_API_KEY)
vectorstore = build_syllabus_vectorstore()
tools = make_tools(vectorstore)
agent = SyllabusAssistantAgent(llm=llm, tools=tools)

app = FastAPI(title="ConnectX Syllabus & Exam Assistant")

# Wide-open CORS for local development against the vanilla-JS frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "syllabus-exam-assistant"}


@app.get("/courses", response_model=list[CourseInfo])
def list_courses():
    seen = {}
    for doc in SYLLABUS_DOCUMENTS:
        code = doc.metadata["course_code"]
        seen[code] = doc.metadata["course_name"]
    return [CourseInfo(course_code=c, course_name=n) for c, n in seen.items()]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    user_input = req.message
    if req.course_code:
        user_input = f"[Course context: {req.course_code}] {user_input}"

    reply = agent.chat(session_id=req.session_id, user_input=user_input)
    return ChatResponse(session_id=req.session_id, response=reply)


@app.delete("/chat/{session_id}")
def reset_session(session_id: str):
    agent.reset(session_id)
    return {"status": "reset", "session_id": session_id}
