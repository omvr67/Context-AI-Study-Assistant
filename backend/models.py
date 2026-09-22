"""Pydantic request/response models for the FastAPI backend.

Same style as the Task/TaskItem models in the Demystifying APIs
notebook: plain BaseModel subclasses with type-hinted fields.

v1.1: added ChatRequest.mode (ELI5 / Exam / Teach-me modes), SourceChunk.page
(page-aware citations for PDF-sourced material), ChatResponse.mode (echoes
back which mode actually applied), and CourseInfo.source/page_count so the
frontend can show a course as PDF- vs text-sourced.

v1.2: added ChatRequest.device_id (identifies which device's private
notebook, if any, this turn's search_notebook tool should be scoped to --
see backend/notebook_store.py and backend/tools.py's make_notebook_tool)
and NotebookDocInfo, the /notebook equivalent of CourseInfo.

Solutions addition: CourseInfo.has_solutions / NotebookDocInfo.has_solutions
report whether an answer-key PDF has been attached via
POST /courses/{course_code}/solutions or POST /notebook/{doc_id}/solutions.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    course_code: str | None = None
    mode: str | None = None  # None/"normal", "eli5", "exam", "teach"
    device_id: str | None = None


class SourceChunk(BaseModel):
    course_code: str
    snippet: str
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    grounded: bool = False
    sources: list[SourceChunk] = []
    tools_used: list[str] = []
    mode: str | None = None


class CourseInfo(BaseModel):
    course_code: str
    course_name: str
    custom: bool = False
    source: str = "text"
    page_count: int | None = None
    has_solutions: bool = False


class AddCourseRequest(BaseModel):
    course_code: str
    course_name: str
    content: str


class NotebookDocInfo(BaseModel):
    doc_id: str
    title: str
    page_count: int
    has_solutions: bool = False


class ResetResponse(BaseModel):
    status: str
    session_id: str
    summary: str | None = None
