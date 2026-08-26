"""
Local persistence for user-added syllabi and uploaded course material.

Courses added through POST /courses (pasted text) or POST /courses/upload
(PDF) live in a small JSON file next to this module -- no database needed
at this scale. On every server restart, `rag.build_syllabus_vectorstore()`
reads this file and re-embeds whatever's here alongside the three
hardcoded syllabi, so anything a student adds sticks around for next time.

Each stored record carries a "source" ("text" or "pdf") and, for PDF
uploads, a "pages" list of {"page": int, "text": str} alongside a
flattened "content" string -- kept for backward compatibility with records
saved before "pages" existed, and so the whole-course text is always
available in one place (e.g. for build_ai_study_plan's topic extraction)
regardless of whether the course came from pasted text or a PDF.
"""
import json
from pathlib import Path

STORE_PATH = Path(__file__).parent / "custom_syllabi.json"


def load_custom_courses() -> list[dict]:
    """Returns [] if nothing has been added yet or the file is unreadable."""
    if not STORE_PATH.exists():
        return []
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_custom_courses(courses: list[dict]) -> None:
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2)


def _find(courses: list[dict], course_code: str) -> dict | None:
    for c in courses:
        if c["course_code"] == course_code:
            return c
    return None


def add_custom_course(course_code: str, course_name: str, content: str) -> None:
    """Adds a pasted-text course, replacing any earlier version saved under the same code."""
    courses = [c for c in load_custom_courses() if c["course_code"] != course_code]
    courses.append({
        "course_code": course_code,
        "course_name": course_name,
        "content": content,
        "source": "text",
        "pages": None,
    })
    save_custom_courses(courses)


def add_pdf_course(course_code: str, course_name: str, pages: list[dict], append: bool = False) -> dict:
    """Saves a PDF-sourced course. If append=True and a course already
    exists under this code with its own pages, the new pages are added
    after the existing ones (so uploading a second chapter under the same
    course code grows the material instead of replacing it); otherwise any
    earlier version under the same code is fully replaced.

    Returns the saved record.
    """
    courses = load_custom_courses()
    existing = _find(courses, course_code) if append else None

    if existing and existing.get("pages"):
        merged_pages = existing["pages"] + pages
    else:
        merged_pages = pages

    record = {
        "course_code": course_code,
        "course_name": course_name,
        "content": "\n\n".join(p["text"] for p in merged_pages),
        "source": "pdf",
        "pages": merged_pages,
    }
    courses = [c for c in courses if c["course_code"] != course_code]
    courses.append(record)
    save_custom_courses(courses)
    return record


def remove_custom_course(course_code: str) -> bool:
    """Returns True if a course was actually removed."""
    courses = load_custom_courses()
    remaining = [c for c in courses if c["course_code"] != course_code]
    if len(remaining) == len(courses):
        return False
    save_custom_courses(remaining)
    return True
