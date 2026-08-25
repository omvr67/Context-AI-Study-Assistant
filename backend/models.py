"""Pydantic request/response models for the FastAPI backend.

Same style as the Task/TaskItem models in the Demystifying APIs
notebook: plain BaseModel subclasses with type-hinted fields.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    course_code: str | None = None


class SourceChunk(BaseModel):
    course_code: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    grounded: bool = False
    sources: list[SourceChunk] = []
    tools_used: list[str] = []


class CourseInfo(BaseModel):
    course_code: str
    course_name: str
    custom: bool = False


class AddCourseRequest(BaseModel):
    course_code: str
    course_name: str
    content: str


class ResetResponse(BaseModel):
    status: str
    session_id: str
    summary: str | None = None
