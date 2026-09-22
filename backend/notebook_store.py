"""
Local persistence for each student's private notebook PDFs.

Unlike custom_courses.py (whose syllabi are visible to every user of this
demo), notebook documents are scoped to a device_id generated client-side
and kept in the browser's localStorage -- each device only ever sees its
own uploads. Same JSON-file-next-to-this-module approach as
custom_courses.py: no database needed at this scale, and it persists
across server restarts. Mirrors custom_courses.py's page-aware PDF storage
so search_notebook can cite "p.N" the same way search_syllabus already
does for uploaded course PDFs.

Storage shape: { "<device_id>": [ {doc_id, title, pages, page_count}, ... ] }

Solutions addition: a doc record may also carry a "solutions" key -- the
same {"page": int, "text": str} page list shape, from a separate
answer-key PDF attached via POST /notebook/{doc_id}/solutions (see main.py
and rag.add_notebook_solutions_to_vectorstore).
"""
import json
import uuid
from pathlib import Path

STORE_PATH = Path(__file__).parent / "notebook_docs.json"


def _load_all() -> dict[str, list[dict]]:
    if not STORE_PATH.exists():
        return {}
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict[str, list[dict]]) -> None:
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_all_notebook_docs() -> dict[str, list[dict]]:
    """Every device's saved notebook docs, keyed by device_id -- used once
    at startup to re-index everything into the live FAISS store after a
    restart. Callers that only care about one device should use
    load_notebook_docs() instead.
    """
    return _load_all()


def load_notebook_docs(device_id: str) -> list[dict]:
    """Returns [] if this device hasn't uploaded anything yet."""
    return _load_all().get(device_id, [])


def add_notebook_doc(device_id: str, title: str, pages: list[dict]) -> dict:
    """Saves one private notebook PDF for a device. Returns the stored
    record (including its generated doc_id) so the caller can index it
    into the vectorstore and track its chunk ids for later removal.
    """
    data = _load_all()
    docs = data.setdefault(device_id, [])
    record = {
        "doc_id": uuid.uuid4().hex[:12],
        "title": title,
        "pages": pages,
        "page_count": len(pages),
    }
    docs.append(record)
    _save_all(data)
    return record


def remove_notebook_doc(device_id: str, doc_id: str) -> bool:
    """Returns True if a document belonging to this device was actually
    removed. A doc_id that only exists under a *different* device_id is
    treated as not found here -- callers should check ownership via
    load_notebook_docs() before calling this, so one device can never
    delete (or learn anything about) another device's upload.
    """
    data = _load_all()
    docs = data.get(device_id, [])
    remaining = [d for d in docs if d["doc_id"] != doc_id]
    if len(remaining) == len(docs):
        return False
    data[device_id] = remaining
    _save_all(data)
    return True


def add_doc_solutions(device_id: str, doc_id: str, pages: list[dict]) -> dict | None:
    """Attaches (or replaces) a solutions/answer-key PDF's pages on one of
    this device's existing notebook docs. Returns the updated record, or
    None if doc_id doesn't exist under this device_id -- checked the same
    ownership-first way remove_notebook_doc is, so one device can't attach
    solutions to (or even confirm the existence of) another device's
    notebook doc.
    """
    data = _load_all()
    docs = data.get(device_id, [])
    for d in docs:
        if d["doc_id"] == doc_id:
            d["solutions"] = pages
            _save_all(data)
            return d
    return None
