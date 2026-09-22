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

Solutions addition: a record may also carry a "solutions" key -- the same
{"page": int, "text": str} page list shape, from a separate answer-key PDF
attached via POST /courses/{course_code}/solutions (see main.py and
rag.add_course_solutions_to_vectorstore). add_custom_course/add_pdf_course
both preserve an existing "solutions" value across a base-material
replace, so re-uploading a corrected syllabus doesn't silently drop an
already-attached answer key.
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


def get_custom_course(course_code: str) -> dict | None:
    """Returns the stored record for one custom (saved) course, or None if
    course_code isn't one -- e.g. it's a built-in, or was never added.
    """
    return _find(load_custom_courses(), course_code)


def add_custom_course(course_code: str, course_name: str, content: str) -> None:
    """Adds a pasted-text course, replacing any earlier version saved under
    the same code. An already-attached solutions PDF, if any, is carried
    over rather than dropped.
    """
    courses = load_custom_courses()
    existing = _find(courses, course_code)
    courses = [c for c in courses if c["course_code"] != course_code]
    record = {
        "course_code": course_code,
        "course_name": course_name,
        "content": content,
        "source": "text",
        "pages": None,
    }
    if existing and existing.get("solutions"):
        record["solutions"] = existing["solutions"]
    courses.append(record)
    save_custom_courses(courses)


def add_pdf_course(course_code: str, course_name: str, pages: list[dict], append: bool = False) -> dict:
    """Saves a PDF-sourced course. If append=True and a course already
    exists under this code with its own pages, the new pages are added
    after the existing ones (so uploading a second chapter under the same
    course code grows the material instead of replacing it); otherwise any
    earlier version under the same code is fully replaced -- except an
    already-attached solutions PDF, which is carried over either way.

    Returns the saved record.
    """
    courses = load_custom_courses()
    existing = _find(courses, course_code)
    prior_for_append = existing if append else None

    if prior_for_append and prior_for_append.get("pages"):
        merged_pages = prior_for_append["pages"] + pages
    else:
        merged_pages = pages

    record = {
        "course_code": course_code,
        "course_name": course_name,
        "content": "\n\n".join(p["text"] for p in merged_pages),
        "source": "pdf",
        "pages": merged_pages,
    }
    if existing and existing.get("solutions"):
        record["solutions"] = existing["solutions"]
    courses = [c for c in courses if c["course_code"] != course_code]
    courses.append(record)
    save_custom_courses(courses)
    return record


def add_course_solutions(course_code: str, pages: list[dict]) -> dict | None:
    """Attaches (or replaces) a solutions/answer-key PDF's pages on an
    existing custom course record. Returns the updated record, or None if
    course_code isn't a saved custom course (callers should check this --
    e.g. main.py's upload endpoint rejects attaching solutions to a
    built-in course, or one that hasn't been added yet).
    """
    courses = load_custom_courses()
    existing = _find(courses, course_code)
    if existing is None:
        return None
    existing["solutions"] = pages
    save_custom_courses(courses)
    return existing


def remove_custom_course(course_code: str) -> bool:
    """Returns True if a course was actually removed."""
    courses = load_custom_courses()
    remaining = [c for c in courses if c["course_code"] != course_code]
    if len(remaining) == len(courses):
        return False
    save_custom_courses(remaining)
    return True
