"""
Local persistence for user-added syllabi.

Courses added through POST /courses live in a small JSON file next to
this module -- no database needed at this scale. On every server
restart, `rag.build_syllabus_vectorstore()` reads this file and
re-embeds whatever's here alongside the three hardcoded syllabi, so
anything a student adds sticks around for next time.

Each stored record is wrapped into the exact same Document shape
(page_content + course_code/course_name metadata) as the hardcoded
syllabi in data.py, so the RAG pipeline can't tell custom syllabi
apart from built-in ones at query time.
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


def add_custom_course(course_code: str, course_name: str, content: str) -> None:
    """Adds a course, replacing any earlier version saved under the same code."""
    courses = [c for c in load_custom_courses() if c["course_code"] != course_code]
    courses.append({
        "course_code": course_code,
        "course_name": course_name,
        "content": content,
    })
    save_custom_courses(courses)


def remove_custom_course(course_code: str) -> bool:
    """Returns True if a course was actually removed."""
    courses = load_custom_courses()
    remaining = [c for c in courses if c["course_code"] != course_code]
    if len(remaining) == len(courses):
        return False
    save_custom_courses(remaining)
    return True
