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

v1.4: added check_notebook_quota(), a per-device cap (NOTEBOOK_MAX_DOCS /
NOTEBOOK_MAX_MB, see backend/config.py) that main.py's /notebook/upload
calls before persisting a new document. Also added a "created_at" field
to every new record and two more functions: evict_expired_notebook_docs()
(DATA_TTL_HOURS cleanup, called by main.py's background sweep) and
clear_device_notebook_docs() (the manual "delete my data" path, DELETE
/me). A record saved before this patch has no "created_at" and is simply
never swept -- see evict_expired_notebook_docs()'s docstring.
"""
import json
import time
import uuid
from pathlib import Path

from .config import NOTEBOOK_MAX_DOCS, NOTEBOOK_MAX_MB

STORE_PATH = Path(__file__).parent / "notebook_docs.json"


class NotebookQuotaError(ValueError):
    """Raised by check_notebook_quota() when adding a document would put a
    device over NOTEBOOK_MAX_DOCS or NOTEBOOK_MAX_MB (see backend/config.py).
    main.py catches this and turns it into a 413.
    """


def check_notebook_quota(device_id: str, new_pages: list[dict]) -> None:
    """Raises NotebookQuotaError if adding a document with `new_pages`
    would put this device over its per-device notebook quota.

    Measured against extracted text size (what's actually persisted here
    and what gets embedded), not the original PDF's byte size, which
    isn't stored in this module at all. Only counts each existing doc's
    main `pages`, not any attached `solutions` pages -- solutions are an
    optional add-on to a doc already under quota, not a second upload the
    quota is meant to gate.

    A demo-scale, per-device-id quota, not a real per-account one: see
    this module's docstring for why device_id (not a real account) is the
    scoping key here.
    """
    existing = load_notebook_docs(device_id)
    if len(existing) >= NOTEBOOK_MAX_DOCS:
        raise NotebookQuotaError(
            f"Your notebook is at its {NOTEBOOK_MAX_DOCS}-document limit for this demo "
            "-- remove one before adding another."
        )

    existing_bytes = sum(
        len(p.get("text", "").encode("utf-8")) for doc in existing for p in doc.get("pages", [])
    )
    new_bytes = sum(len(p.get("text", "").encode("utf-8")) for p in new_pages)
    max_bytes = NOTEBOOK_MAX_MB * 1024 * 1024
    if existing_bytes + new_bytes > max_bytes:
        raise NotebookQuotaError(
            f"Your notebook is at its {NOTEBOOK_MAX_MB} MB limit for this demo "
            "-- remove something before adding more."
        )


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
        "created_at": time.time(),
    }
    docs.append(record)
    _save_all(data)
    return record


def evict_expired_notebook_docs(ttl_seconds: float) -> list[dict]:
    """Removes every notebook doc, across every device, whose
    "created_at" is older than ttl_seconds. Persists the change and
    returns the removed records (each with "device_id" merged in) so the
    caller -- main.py's background sweep -- can also drop their
    vectorstore chunks; this module has no notion of the FAISS index at
    all (same separation custom_courses.py's docstring describes).

    A record with no "created_at" (saved before this field existed) is
    always kept, never evicted -- "unknown age" isn't "expired", and
    guessing an age for old data would risk deleting something still in
    active use the moment this patch is deployed.
    """
    if ttl_seconds <= 0:
        return []
    cutoff = time.time() - ttl_seconds
    data = _load_all()
    removed: list[dict] = []
    changed = False
    for device_id, docs in list(data.items()):
        keep = []
        for d in docs:
            created_at = d.get("created_at")
            if created_at is not None and created_at < cutoff:
                removed.append({**d, "device_id": device_id})
                changed = True
            else:
                keep.append(d)
        if keep:
            data[device_id] = keep
        else:
            data.pop(device_id, None)
    if changed:
        _save_all(data)
    return removed


def clear_device_notebook_docs(device_id: str) -> list[dict]:
    """Removes every notebook doc for one device_id outright -- the
    manual "delete my data" path (DELETE /me in main.py), for a visitor
    who doesn't want to wait for the DATA_TTL_HOURS sweep. Returns the
    removed records so the caller can drop their vectorstore chunks too.
    """
    data = _load_all()
    docs = data.pop(device_id, None) or []
    if docs:
        _save_all(data)
    return docs


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
