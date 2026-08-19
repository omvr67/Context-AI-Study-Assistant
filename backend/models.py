"""Pydantic request/response models for the FastAPI backend.

Same style as the Task/TaskItem models in the Demystifying APIs
notebook: plain BaseModel subclasses with type-hinted fields.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    course_code: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


class CourseInfo(BaseModel):
    course_code: str
    course_name: str
