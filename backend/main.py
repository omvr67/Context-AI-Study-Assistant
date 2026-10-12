"""
FastAPI backend for the Syllabus & Exam Assistant.

Endpoint shapes follow the Demystifying APIs notebook conventions:
Pydantic request/response models, HTTPException for error cases, and a
plain FastAPI() app instance. Run from the project root with:

 To actiate:
   uvicorn backend.main:app --reload --port 8000

v1.1 additions: POST /courses/upload (PDF ingestion) and POST /chat/stream
(SSE streaming chat) -- see their docstrings below for details.

v1.3 addition: GET/POST/DELETE /notebook* -- a private, per-device PDF
notebook separate from the shared course library above (see
backend/notebook_store.py). ChatRequest.device_id, when present, binds a
request-scoped search_notebook tool for that one call (see
backend/tools.py's make_notebook_tool) so the agent can search a device's
own notebook without ever being able to search anyone else's.

Solutions addition: POST /courses/{course_code}/solutions and
POST /notebook/{doc_id}/solutions attach an answer-key PDF to an existing
course or notebook doc. Indexed under the same course_code/doc_id as the
source material (so the existing search_syllabus/search_notebook calls
retrieve both together) but tagged material="solutions" so citations --
and the agent's reasoning -- can tell them apart (see rag.format_docs).
course_solutions_chunk_ids / notebook_solutions_chunk_ids track these
chunks separately from each course's/doc's own, so a re-upload can replace
just the solutions instead of duplicating them.

Rate-limiting addition: every endpoint that calls the embedder or the LLM
is throttled per visitor via backend/rate_limit.py (in-memory, two
windows -- RATE_LIMIT_PER_MINUTE and RATE_LIMIT_PER_DAY, see
backend/config.py) before any expensive work happens, and every Groq call
inside the agent retries with backoff on a 429 -- see backend/agent.py.

Flashcards addition: POST /flashcards, a bare "/flashcards" command
(typed into the composer, not a button -- see frontend/app.js) that
generates a 10-card deck from what's actually been discussed in that chat
session. Deliberately its own endpoint rather than a chat tool: it never
touches the agent's tool-calling loop, only session_id to pull the
session's transcript -- see agent.py's recent_transcript() /
generate_flashcards_from_conversation() for the one-shot structured-output
call backing it. (v1.3 round 2: this replaced an earlier version that
generated a deck from a resolved course's/notebook doc's raw material
instead of the conversation.)

v1.3 round 2 addition: ChatRequest.notebook_doc_ids lets a chat turn be
pinned to exactly the notebook PDF(s) the student selected in the
sidebar -- see _notebook_scope() just above the /chat endpoints, which
resolves that into a doc_id-scoped search_notebook tool plus (when any
doc is pinned) excludes search_syllabus/build_ai_study_plan from the
toolset entirely and adds a short system note telling the model
search_notebook is now its only grounding source. Also removed the
eli5/teach modes and the mode-bar UI entirely, frontend and backend --
ChatRequest.mode now only ever carries "depth" or "exam", both one-shot
"/depth"/"/exam" commands typed into the composer (see agent.py's
MODE_PROMPTS).

Quiz addition (v1.3 round 3): POST /quiz, same /command-not-a-chat-message
shape as /flashcards above, generates a 5-question multiple-choice quiz
from the session's conversation (agent.py's
generate_quiz_from_conversation()). Grading happens client-side the
instant the student picks an answer -- POST /quiz/results just records
the outcome server-side (agent.py's record_quiz_results()) once a quiz is
finished, so POST /retry-quiz can generate a follow-up quiz targeting
whatever that session has actually gotten wrong
(generate_retry_quiz()) -- the real graded-quiz version of the
originally-requested "retry exam" feature, in place of inferring
correctness from free-form chat. DELETE /chat/{session_id} (reset)
clears a session's quiz history alongside its chat history, so a retry
quiz always draws on the conversation actually still in front of the
student.

v1.4 (production-readiness round, delivered as several small stacked
patches rather than one large one): starting with config foundation --
backend/config.py centralizes the environment-driven settings this round
introduces (model name, demo-mode flag, admin token, CORS origins), each
with a default that reproduces pre-v1.4 behavior so an existing local
`.env` with just GROQ_API_KEY keeps working unchanged. CORS now reads
ALLOWED_ORIGINS instead of a hardcoded wildcard, and the Groq model name
is configurable via GROQ_MODEL instead of hardcoded. Later patches in
this round (same file, same config module) add: visitor-data TTL
eviction, request timeouts/concurrency limits, and structured logging.

Admin-token protection (also v1.4): the four shared-course write
endpoints (add by text, add by PDF, attach solutions, delete) now call
_require_admin first. It's a no-op while DEMO_MODE is off, so this has
zero effect locally. Once a public demo turns DEMO_MODE on, the shared
course library -- which every visitor reads -- becomes read-only to
everyone except whoever's browser holds ADMIN_TOKEN (see
frontend/app.js's getOrSetAdminToken: visiting the deployed link once
with ?admin=<token> stores it in that browser's localStorage and
immediately strips it from the address bar). A visitor's own private
notebook (/notebook/*) is untouched by this -- it was never shared
across visitors in the first place.

Rate-limit hardening (also v1.4): _client_key now prefers a
proxy-set header over request.client.host, because behind a tunnel
(ngrok, Cloudflare Tunnel, ...) every visitor's request arrives at this
process from the tunnel daemon on localhost -- request.client.host alone
would be identical for every visitor, the exact "one heavy tester locks
everyone out" failure rate_limit.py exists to prevent. See its docstring
for why trusting that header is safe specifically in this setup. The
throttle itself now has a second, daily window (RATE_LIMIT_PER_DAY) on
top of the existing per-minute one, and is now enforced -- via the new
_enforce_rate_limit helper -- on every endpoint that calls the embedder
or the LLM, not just /chat and /chat/stream: the course/notebook
upload-and-embed endpoints, /flashcards, /quiz, /retry-quiz, and a
session reset that asks for a summary.

Same-origin frontend addition: this file now serves frontend/ itself
(StaticFiles mount at "/", added last so it only ever catches a request
none of the API routes above already matched) instead of the frontend
being opened separately via file:// or a second dev server. That's what
makes the tunnel-shared demo actually work -- a browser refuses to let an
https tunnel page call a plain http API, and frontend/app.js's API_BASE
is now a relative "" for exactly that reason. One visible side effect:
the old GET "/" health check moved to GET "/health" so "/" is free for
index.html; same response shape, nothing else about it changed yet (a
later patch in this round fleshes it out further).

Input/resource caps (also v1.4): /chat and /chat/stream now reject a
message over MAX_MESSAGE_LENGTH (422); agent.py's _history() trims each
session to its last HISTORY_WINDOW_TURNS turns -- see that method's own
docstring. Every PDF upload (course and notebook, main doc and
solutions) now reads through _read_upload_capped instead of a one-shot
file.read(), so an oversized body is rejected the moment it exceeds
MAX_PDF_BYTES on the wire rather than after being pulled fully into
memory first. /notebook/upload additionally checks
notebook_store.check_notebook_quota (NOTEBOOK_MAX_DOCS / NOTEBOOK_MAX_MB
per device) before persisting.

Data lifecycle (also v1.4): a background task (_sweep_loop, started by
_lifespan at app startup) wakes up every 15 minutes and evicts chat/quiz
sessions idle longer than DATA_TTL_HOURS (agent.evict_expired_sessions)
and notebook documents uploaded more than DATA_TTL_HOURS ago
(notebook_store.evict_expired_notebook_docs), purging the latter's
vectorstore chunks via the new _purge_notebook_doc_index -- now also
shared by DELETE /notebook/{doc_id}. DELETE /me is the manual
counterpart: a visitor can clear their own notebook and chat/quiz session
immediately rather than waiting on the sweep.

Resilience (also v1.4): the ChatGroq client is now constructed with
timeout=LLM_CALL_TIMEOUT_SECONDS, bounding every individual HTTP request
it makes. Separately, agent.py's _invoke_with_backoff/_stream_with_backoff
now cap how many Groq calls run at once (MAX_CONCURRENT_LLM_CALLS) and
raise LLMBusyError if a call can't get a slot within
LLM_QUEUE_WAIT_SECONDS -- /chat and /chat/stream turn that into a normal
assistant-message reply (agent.py's BUSY_MESSAGE), the same way they
already do for a 429; /flashcards, /quiz, and /retry-quiz turn it into a
503 here.

Health + logging (also v1.4, closing out this round): logging.basicConfig
is configured once at import time (LOG_LEVEL, see backend/config.py),
replacing the sweep's old print() calls with real logger calls and adding
a few more: once at startup (model/demo_mode/rate-limit/TTL settings, so
a launch's actual config is visible in the console it's running in), and
on a rate-limit or admin-check rejection (the client key, never the
supplied admin token itself). _log_requests is a middleware logging one
line per request -- method, path, status, duration -- tagged with a
short request id that's also echoed back as the X-Request-ID response
header, skipping static assets so the console stays readable during a
live demo. GET /health now reports uptime, demo_mode, the model in use,
the live chat-session count, and llm_calls_in_flight() /
llm_concurrency_limit (see agent.py) for at-a-glance load visibility.
"""
import asyncio
import contextlib
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # connects the front and back ends in browser
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq

from .agent import (
    BUSY_MESSAGE,
    FRIENDLY_QUOTA_MESSAGE,
    LLMBusyError,
    SyllabusAssistantAgent,
    _is_rate_limit_error,
    llm_calls_in_flight,
)
from .config import (
    ADMIN_TOKEN,
    ALLOWED_ORIGINS,
    DATA_TTL_HOURS,
    DEMO_MODE,
    GROQ_MODEL,
    LLM_CALL_TIMEOUT_SECONDS,
    LOG_LEVEL,
    MAX_CONCURRENT_LLM_CALLS,
    MAX_MESSAGE_LENGTH,
    RATE_LIMIT_PER_DAY,
    RATE_LIMIT_PER_MINUTE,
)
from .custom_courses import (
    add_course_solutions,
    add_custom_course,
    add_pdf_course,
    get_custom_course,
    load_custom_courses,
    remove_custom_course,
)
from .data import SYLLABUS_DOCUMENTS
from .models import (
    AddCourseRequest,
    ChatRequest,
    ChatResponse,
    CourseInfo,
    Flashcard,
    FlashcardDeck,
    FlashcardsRequest,
    NotebookDocInfo,
    QuizDeck,
    QuizQuestion,
    QuizRequest,
    QuizResultsRequest,
    ResetResponse,
    SourceChunk,
)
from .notebook_store import (
    NotebookQuotaError,
    add_doc_solutions,
    add_notebook_doc,
    check_notebook_quota,
    clear_device_notebook_docs,
    evict_expired_notebook_docs,
    load_all_notebook_docs,
    load_notebook_docs,
    remove_notebook_doc,
)
from .pdf_ingest import MAX_PDF_BYTES, PDFValidationError, extract_pdf_pages
from .rag import (
    add_course_solutions_to_vectorstore,
    add_course_to_vectorstore,
    add_notebook_doc_to_vectorstore,
    add_notebook_solutions_to_vectorstore,
    add_pdf_course_to_vectorstore,
    build_syllabus_vectorstore,
    remove_course_from_vectorstore,
)
from .rate_limit import check_and_increment
from .tools import make_notebook_tool, make_tools  # imports functions that create the tools used

# ---------------------------------------------------------------------------
# Logging: plain text to stdout, one line per request (see _log_requests
# below) plus the handful of explicit logger.* calls scattered through
# this file (startup, the data-TTL sweep, rate-limit/admin rejections).
# No file rotation, no JSON formatter, no third-party log shipper -- this
# is a single demo process, not a fleet; `uvicorn ... 2>&1 | tee demo.log`
# is enough to keep a copy if the terminal itself isn't enough.
# ---------------------------------------------------------------------------
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("connectx")

_START_TIME = time.monotonic()

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
    model=GROQ_MODEL,
    temperature=0.0,
    max_tokens=8192, #Increase the max tokens to 8192 for larger context
    api_key=GROQ_API_KEY,
    timeout=LLM_CALL_TIMEOUT_SECONDS,  # bounds every HTTP request this client makes to Groq
)

# vectorstore starts loaded with the 3 hardcoded syllabi *and* anything a
# student has already saved locally via POST /courses or /courses/upload on
# a previous run. course_solutions_chunk_ids tracks any attached answer-key
# PDFs separately from each course's own material -- see
# rag.build_syllabus_vectorstore's docstring for why.
vectorstore, course_chunk_ids, course_solutions_chunk_ids = build_syllabus_vectorstore()

# Private per-device notebook: re-index every device's saved uploads (see
# backend/notebook_store.py) into this same live FAISS store, tagged with
# device_id + notebook=True so they can only ever surface through
# search_notebook's own metadata filter -- never through search_syllabus,
# and never for a device other than the one that uploaded them.
# notebook_solutions_chunk_ids is the notebook-side twin of
# course_solutions_chunk_ids above, same reason: tracked separately so a
# re-upload can replace just the solutions chunks for one doc_id.
notebook_chunk_ids: dict[str, list[str]] = {}  # doc_id -> chunk ids
notebook_solutions_chunk_ids: dict[str, list[str]] = {}  # doc_id -> solutions chunk ids
for _device_id, _docs in load_all_notebook_docs().items():
    for _doc in _docs:
        notebook_chunk_ids[_doc["doc_id"]] = add_notebook_doc_to_vectorstore(
            vectorstore, _device_id, _doc["doc_id"], _doc["title"], _doc["pages"]
        )
        _solutions_pages = _doc.get("solutions")
        if _solutions_pages:
            notebook_solutions_chunk_ids[_doc["doc_id"]] = add_notebook_solutions_to_vectorstore(
                vectorstore, _device_id, _doc["doc_id"], _doc["title"], _solutions_pages
            )

tools = make_tools(vectorstore)
agent = SyllabusAssistantAgent(llm=llm, tools=tools)

logger.info(
    "ConnectX starting: model=%s demo_mode=%s admin_token_set=%s "
    "rate_limit=%d/min,%d/day data_ttl=%dh max_concurrent_llm=%d courses_indexed=%d",
    GROQ_MODEL, DEMO_MODE, ADMIN_TOKEN is not None,
    RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_DAY, DATA_TTL_HOURS, MAX_CONCURRENT_LLM_CALLS,
    len(course_chunk_ids),
)

BASE_COURSE_CODES = {doc.metadata["course_code"] for doc in SYLLABUS_DOCUMENTS}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

DATA_TTL_SECONDS = DATA_TTL_HOURS * 3600
_SWEEP_INTERVAL_SECONDS = 15 * 60  # how often the background task checks, not the TTL itself


def _purge_notebook_doc_index(doc_id: str) -> None:
    """Drops one notebook doc's vectorstore chunks (main content and, if
    attached, solutions) from the live FAISS index and from the chunk-id
    tracking dicts. Shared by DELETE /notebook/{doc_id}, DELETE /me, and
    the DATA_TTL_HOURS sweep below -- every place a notebook doc's
    underlying *record* has already been removed from notebook_store.py
    and just needs its index entries cleaned up to match.
    """
    ids = notebook_chunk_ids.pop(doc_id, None)
    if ids:
        remove_course_from_vectorstore(vectorstore, ids)
    solutions_ids = notebook_solutions_chunk_ids.pop(doc_id, None)
    if solutions_ids:
        remove_course_from_vectorstore(vectorstore, solutions_ids)


def _sweep_expired_data() -> None:
    """One sweep pass: evicts chat/quiz sessions idle longer than
    DATA_TTL_HOURS (agent.evict_expired_sessions) and notebook documents
    uploaded more than DATA_TTL_HOURS ago (evict_expired_notebook_docs),
    purging the latter's vectorstore chunks too. Logged on stdout when it
    actually does something, so an operator watching the demo server's
    console can see the sweep is really running rather than just trusting
    that it will.
    """
    evicted_sessions = agent.evict_expired_sessions(DATA_TTL_SECONDS)
    expired_docs = evict_expired_notebook_docs(DATA_TTL_SECONDS)
    for doc in expired_docs:
        _purge_notebook_doc_index(doc["doc_id"])
    if evicted_sessions or expired_docs:
        logger.info("data-ttl sweep: evicted %d session(s), %d notebook doc(s)", evicted_sessions, len(expired_docs))


async def _sweep_loop() -> None:
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
        try:
            _sweep_expired_data()
        except Exception as e:  # a sweep bug should never take the server down
            logger.warning("data-ttl sweep failed: %s", e)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Starts the DATA_TTL_HOURS background sweep alongside the app and
    cancels it on shutdown. DATA_TTL_HOURS <= 0 disables the sweep
    entirely (no task started) rather than spinning in a loop that would
    never find anything to evict.
    """
    task = asyncio.create_task(_sweep_loop()) if DATA_TTL_SECONDS > 0 else None
    yield
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="ConnectX Syllabus & Exam Assistant", lifespan=_lifespan)

# ALLOWED_ORIGINS defaults to "*" (today's behavior) when unset -- fine for
# local dev, but set it to the real tunnel/deploy origin in .env before
# sharing a public link. See backend/config.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# One access-log line per request, each tagged with a short request id
# (also echoed back as the X-Request-ID response header, so a specific
# browser request can be matched to a specific server log line while
# watching a live demo). Skips static assets (style.css, app.js, the
# vendored libraries, fonts, ...) served by the StaticFiles mount at the
# bottom of this file -- every real API route below has no file extension
# in its path, which is what the check below actually keys on, so this
# needs no separate list of routes to stay in sync with.
_STATIC_EXTENSIONS = (".css", ".js", ".html", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2")


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    if request.url.path.lower().endswith(_STATIC_EXTENSIONS):
        return await call_next(request)

    request_id = uuid.uuid4().hex[:8]
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s %s -> %d (%.0fms)",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


def _course_registry() -> dict[str, dict]:
    """course_code -> {course_name, custom, source, page_count, has_solutions}
    for every indexed course, base + custom.
    """
    registry: dict[str, dict] = {}
    for doc in SYLLABUS_DOCUMENTS:
        registry[doc.metadata["course_code"]] = {
            "course_name": doc.metadata["course_name"],
            "custom": False,
            "source": "text",
            "page_count": None,
            "has_solutions": False,
        }
    for course in load_custom_courses():
        pages = course.get("pages")
        registry[course["course_code"]] = {
            "course_name": course["course_name"],
            "custom": True,
            "source": course.get("source", "text"),
            "page_count": len(pages) if pages else None,
            "has_solutions": bool(course.get("solutions")),
        }
    return registry


def _client_key(request: Request) -> str:
    """Best-effort visitor identifier for the rate limiter.

    Behind a tunnel (ngrok, Cloudflare Tunnel, localtunnel, ...) every
    request reaches this process from the tunnel daemon on localhost, so
    request.client.host alone is identical for every visitor -- exactly
    the failure mode rate_limit.py exists to prevent. Trusting a
    forwarded-IP header is normally risky (a client could just set it
    themselves), but that's not a realistic path here: nothing external
    can reach this process directly, only the local tunnel daemon can, so
    whatever header it sets is the one piece of information actually
    coming from outside.

    Prefers Cloudflare's CF-Connecting-IP (set by Cloudflare's own edge,
    not by the tunnel binary, so it's the hardest of the three to spoof),
    then the first hop of a standard X-Forwarded-For chain (what ngrok
    and localtunnel both set), then falls back to request.client.host
    unchanged -- today's exact behavior -- for plain local development
    with no proxy in front at all.
    """
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip

    forwarded = request.headers.get("x-forwarded-for", "")
    first_hop = forwarded.split(",")[0].strip()
    if first_hop:
        return first_hop

    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request, action: str = "requests") -> None:
    """Shared throttle check for every endpoint that calls the embedder or
    the LLM -- see backend/rate_limit.py for the actual windows. Raises a
    429 naming `action` (e.g. "messages", "quiz requests") if this
    visitor has exceeded either window; otherwise records the request and
    returns normally.
    """
    key = _client_key(request)
    allowed, retry_after = check_and_increment(key)
    if not allowed:
        logger.warning("rate limit: %s blocked for %ds (%s)", key, retry_after, action)
        raise HTTPException(
            status_code=429,
            detail=f"You're sending {action} a little fast -- please wait {retry_after}s and try again.",
        )


def _require_admin(request: Request) -> None:
    """Guards shared-course writes (add/upload/attach-solutions/delete)
    while DEMO_MODE is on. A no-op when DEMO_MODE is off, so local
    development is completely unaffected by this check -- same principle
    as every other v1.4 setting (see backend/config.py).

    No ADMIN_TOKEN configured means no admin access at all (shared writes
    locked outright) rather than failing open, per DEMO_MODE's own
    docstring -- that's a 403, since no token anyone could send would
    help. A configured token that the request doesn't present or gets
    wrong is a 401, since supplying the right one *would* help. Token
    comparison is constant-time (secrets.compare_digest) so response
    timing can't be used to guess it character by character.
    """
    if not DEMO_MODE:
        return
    if ADMIN_TOKEN is None:
        raise HTTPException(status_code=403, detail="Shared-course writes are locked for this demo.")
    supplied = request.headers.get("x-admin-token", "")
    if not secrets.compare_digest(supplied, ADMIN_TOKEN):
        # Never log the supplied value itself -- only that an attempt failed.
        logger.warning("admin check failed: %s %s from %s", request.method, request.url.path, _client_key(request))
        raise HTTPException(status_code=401, detail="Missing or invalid admin token.")


async def _read_upload_capped(file: UploadFile, max_bytes: int = MAX_PDF_BYTES) -> bytes:
    """Reads an uploaded file in bounded 1 MB chunks, raising 413 the
    instant the real byte count exceeds max_bytes.

    `await file.read()` (the one-shot version used at every upload site
    before this) pulls the *entire* body into memory first and only
    rejects it afterward, via pdf_ingest.py's own MAX_PDF_BYTES check
    inside validate_pdf_header -- so an oversized upload still costs full
    memory for however long that takes. A client-declared Content-Length
    header isn't trusted here either (trivial to omit or understate); this
    reads real bytes off the wire and stops the moment they exceed the
    cap, so it's the one check that actually bounds memory regardless of
    what the request claims about itself.
    """
    chunks = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large (max {max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _require_device_id(device_id: str | None) -> str:
    """Every /notebook endpoint needs a real device_id -- there's no auth
    system here, so this client-generated id (see frontend/app.js) is the
    entire basis for keeping one device's private uploads away from every
    other device. An empty one is rejected outright rather than silently
    falling back to some shared bucket.
    """
    device_id = (device_id or "").strip()
    if not device_id:
        raise HTTPException(status_code=422, detail="device_id is required")
    return device_id


# Backend security check. Moved off "/" in v1.4 so the frontend's
# index.html can be served there instead (see the StaticFiles mount at the
# bottom of this file) -- a later v1.4 patch extends this response.
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "syllabus-exam-assistant",
        "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
        "demo_mode": DEMO_MODE,
        "model": GROQ_MODEL,
        "courses_indexed": len(course_chunk_ids),
        "notebook_docs_indexed": len(notebook_chunk_ids),
        "courses_with_solutions": len(course_solutions_chunk_ids),
        "notebook_docs_with_solutions": len(notebook_solutions_chunk_ids),
        "active_chat_sessions": len(agent._sessions),
        "llm_calls_in_flight": llm_calls_in_flight(),
        "llm_concurrency_limit": MAX_CONCURRENT_LLM_CALLS,
    }


@app.get("/courses", response_model=list[CourseInfo])
def list_courses():
    registry = _course_registry()
    return [
        CourseInfo(
            course_code=code,
            course_name=info["course_name"],
            custom=info["custom"],
            source=info["source"],
            page_count=info["page_count"],
            has_solutions=info["has_solutions"],
        )
        for code, info in registry.items()
    ]


@app.post("/courses", response_model=CourseInfo, status_code=201)
def add_course(req: AddCourseRequest, request: Request):
    code = req.course_code.strip().upper()
    name = req.course_name.strip()
    content = req.content.strip()

    if not code or not name or not content:
        raise HTTPException(status_code=422, detail="course_code, course_name, and content are all required")

    _require_admin(request)
    _enforce_rate_limit(request, "course uploads")
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

    return CourseInfo(course_code=code, course_name=name, custom=True, source="text")


@app.post("/courses/upload", response_model=CourseInfo, status_code=201)
async def upload_course_pdf(
    request: Request,
    course_code: str = Form(...),
    course_name: str = Form(...),
    append: bool = Form(False),
    file: UploadFile = File(...),
):
    """Uploads a PDF syllabus/chapter, validates it, extracts page-aware
    text, and adds it to the live FAISS index so it becomes usable by the
    existing RAG (search_syllabus, build_ai_study_plan, etc.) immediately --
    no restart needed.

    append=True adds the new pages after any existing pages already saved
    under this course_code (growing the material, e.g. uploading chapter 2
    after chapter 1). append=False (default) replaces any earlier version
    under this code entirely, matching how the pasted-text POST /courses
    already behaves.
    """
    code = course_code.strip().upper()
    name = course_name.strip()

    if not code or not name:
        raise HTTPException(status_code=422, detail="course_code and course_name are required")
    if code in BASE_COURSE_CODES:
        raise HTTPException(status_code=409, detail=f"'{code}' is a built-in course and can't be overwritten")

    _require_admin(request)
    _enforce_rate_limit(request, "course uploads")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await _read_upload_capped(file)
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not append and code in course_chunk_ids:
        remove_course_from_vectorstore(vectorstore, course_chunk_ids.pop(code, []))

    new_ids = add_pdf_course_to_vectorstore(vectorstore, code, name, pages)
    if append and code in course_chunk_ids:
        course_chunk_ids[code] = course_chunk_ids[code] + new_ids
    else:
        course_chunk_ids[code] = new_ids

    saved = add_pdf_course(code, name, pages, append=append)
    total_pages = len(saved["pages"])

    return CourseInfo(course_code=code, course_name=name, custom=True, source="pdf", page_count=total_pages)


@app.delete("/courses/{course_code}")
def delete_course(course_code: str, request: Request):
    code = course_code.strip().upper()
    if code in BASE_COURSE_CODES:
        raise HTTPException(status_code=403, detail="Built-in courses can't be deleted")

    _require_admin(request)

    ids = course_chunk_ids.pop(code, None)
    solutions_ids = course_solutions_chunk_ids.pop(code, None)
    removed_from_disk = remove_custom_course(code)
    if ids is None and solutions_ids is None and not removed_from_disk:
        raise HTTPException(status_code=404, detail=f"No custom course '{code}' found")

    if ids:
        remove_course_from_vectorstore(vectorstore, ids)
    if solutions_ids:
        remove_course_from_vectorstore(vectorstore, solutions_ids)
    return {"status": "deleted", "course_code": code}


@app.post("/courses/{course_code}/solutions", response_model=CourseInfo, status_code=201)
async def upload_course_solutions(course_code: str, request: Request, file: UploadFile = File(...)):
    """Attaches a solutions/answer-key PDF to an existing custom course.
    Indexed under the same course_code as the course's own material, but
    tagged material="solutions" so search_syllabus's citations -- and the
    agent's own reasoning -- can tell the two apart even though a search
    retrieves them together (see rag.format_docs and
    GUARDRAIL_SYSTEM_PROMPT's rule 3). Re-uploading replaces any
    previously-attached solutions rather than stacking duplicates: any
    chunks already tracked under this course_code in
    course_solutions_chunk_ids are removed first.

    Only works on a *custom* (already-added) course -- there's nothing
    meaningful to attach an answer key to on a built-in course's plain
    grading-policy syllabus.
    """
    code = course_code.strip().upper()
    if code in BASE_COURSE_CODES:
        raise HTTPException(
            status_code=409, detail=f"'{code}' is a built-in course and can't take an attached solutions PDF"
        )
    existing = get_custom_course(code)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No custom course '{code}' found -- add the course itself first")

    _require_admin(request)
    _enforce_rate_limit(request, "course uploads")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await _read_upload_capped(file)
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if code in course_solutions_chunk_ids:
        remove_course_from_vectorstore(vectorstore, course_solutions_chunk_ids.pop(code, []))

    new_ids = add_course_solutions_to_vectorstore(vectorstore, code, existing["course_name"], pages)
    course_solutions_chunk_ids[code] = new_ids
    add_course_solutions(code, pages)

    info = _course_registry()[code]
    return CourseInfo(
        course_code=code,
        course_name=info["course_name"],
        custom=True,
        source=info["source"],
        page_count=info["page_count"],
        has_solutions=True,
    )


# --- Private per-device notebook -------------------------------------------
# Separate from the shared course library above: a device_id generated and
# stored client-side (see frontend/app.js) scopes every one of these
# endpoints, and the same id is threaded through to make_notebook_tool at
# chat time so an uploaded PDF is only ever searchable by the device that
# uploaded it -- see backend/notebook_store.py and backend/tools.py.

@app.get("/notebook", response_model=list[NotebookDocInfo])
def list_notebook(device_id: str = Query(...)):
    device_id = _require_device_id(device_id)
    return [
        NotebookDocInfo(
            doc_id=d["doc_id"],
            title=d["title"],
            page_count=d.get("page_count", len(d.get("pages", []))),
            has_solutions=bool(d.get("solutions")),
        )
        for d in load_notebook_docs(device_id)
    ]


@app.post("/notebook/upload", response_model=NotebookDocInfo, status_code=201)
async def upload_notebook_pdf(
    request: Request,
    device_id: str = Form(...),
    title: str = Form(...),
    file: UploadFile = File(...),
):
    """Uploads a private PDF to the caller's own notebook. Indexed into the
    same live FAISS store the shared courses use, but tagged with this
    device_id so it's only ever found by a search_notebook call made on
    this device's behalf -- never by search_syllabus, and never by another
    device's notebook search. No restart needed; it's searchable immediately.
    """
    device_id = _require_device_id(device_id)
    doc_title = title.strip()
    if not doc_title:
        raise HTTPException(status_code=422, detail="title is required")

    _enforce_rate_limit(request, "uploads")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await _read_upload_capped(file)
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        check_notebook_quota(device_id, pages)
    except NotebookQuotaError as e:
        raise HTTPException(status_code=413, detail=str(e))

    saved = add_notebook_doc(device_id, doc_title, pages)
    notebook_chunk_ids[saved["doc_id"]] = add_notebook_doc_to_vectorstore(
        vectorstore, device_id, saved["doc_id"], doc_title, pages
    )

    return NotebookDocInfo(doc_id=saved["doc_id"], title=doc_title, page_count=saved["page_count"])


@app.delete("/notebook/{doc_id}")
def delete_notebook_doc(doc_id: str, device_id: str = Query(...)):
    device_id = _require_device_id(device_id)

    # Ownership check happens against the store itself, before touching
    # anything else -- a doc_id that exists but belongs to a different
    # device_id is treated identically to one that doesn't exist at all,
    # so this response can never confirm or deny another device's upload.
    owned_ids = {d["doc_id"] for d in load_notebook_docs(device_id)}
    if doc_id not in owned_ids:
        raise HTTPException(status_code=404, detail=f"No notebook document '{doc_id}' found")

    remove_notebook_doc(device_id, doc_id)
    _purge_notebook_doc_index(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.delete("/me")
def delete_my_data(device_id: str = Query(...), session_id: str = Query(...)):
    """Manual "delete my data": clears this device's entire private
    notebook and this browser's chat/quiz session immediately, instead of
    waiting for the DATA_TTL_HOURS sweep. Both ids are exactly what the
    frontend already holds (device_id in localStorage, session_id
    in-memory for the current tab) -- there's no account system to check
    either against beyond that, same as every other notebook/chat
    endpoint in this file.
    """
    device_id = _require_device_id(device_id)
    removed_docs = clear_device_notebook_docs(device_id)
    for doc in removed_docs:
        _purge_notebook_doc_index(doc["doc_id"])
    agent.reset(session_id)
    return {"status": "deleted", "notebook_docs_removed": len(removed_docs)}


@app.post("/notebook/{doc_id}/solutions", response_model=NotebookDocInfo, status_code=201)
async def upload_notebook_solutions(
    doc_id: str, request: Request, device_id: str = Form(...), file: UploadFile = File(...)
):
    """Notebook equivalent of POST /courses/{course_code}/solutions: attaches
    a solutions/answer-key PDF to one of the caller's own notebook docs.
    Ownership is checked the same way delete_notebook_doc checks it -- a
    doc_id belonging to a different device_id is treated as not found, so
    this can never confirm or attach anything to another device's upload.
    Re-uploading replaces any previously-attached solutions for this doc_id
    rather than stacking duplicates.
    """
    device_id = _require_device_id(device_id)

    owned = {d["doc_id"]: d for d in load_notebook_docs(device_id)}
    existing = owned.get(doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No notebook document '{doc_id}' found")

    _enforce_rate_limit(request, "uploads")

    filename = (file.filename or "").lower()
    if file.content_type not in ("application/pdf", "application/x-pdf") and not filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    data = await _read_upload_capped(file)
    try:
        pages = extract_pdf_pages(data)
    except PDFValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if doc_id in notebook_solutions_chunk_ids:
        remove_course_from_vectorstore(vectorstore, notebook_solutions_chunk_ids.pop(doc_id, []))

    new_ids = add_notebook_solutions_to_vectorstore(vectorstore, device_id, doc_id, existing["title"], pages)
    notebook_solutions_chunk_ids[doc_id] = new_ids
    add_doc_solutions(device_id, doc_id, pages)

    return NotebookDocInfo(
        doc_id=doc_id,
        title=existing["title"],
        page_count=existing.get("page_count", len(existing.get("pages", []))),
        has_solutions=True,
    )


# --- Flashcards -------------------------------------------------------------
# A /command, not a chat message: frontend/app.js intercepts a bare
# "/flashcards" in the composer before it would ever reach POST /chat -- so
# this endpoint never sees ChatRequest or the agent's tool loop. v1.3 round
# 2 reworked this from "resolve a named course/notebook target to its raw
# material" to "read what's actually been discussed in this chat session
# and build a deck around that topic" -- see agent.py's recent_transcript()
# and generate_flashcards_from_conversation().


@app.post("/flashcards", response_model=FlashcardDeck)
def generate_flashcards_endpoint(req: FlashcardsRequest, request: Request):
    transcript = agent.recent_transcript(req.session_id)
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Nothing to build flashcards from yet -- chat about a topic first, then try /flashcards again.",
        )

    _enforce_rate_limit(request, "flashcard requests")

    try:
        topic, cards = agent.generate_flashcards_from_conversation(transcript, count=10)
    except LLMBusyError:
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
    except Exception as e:
        detail = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Couldn't generate flashcards: {e}"
        raise HTTPException(status_code=502, detail=detail)

    if not cards or not topic:
        raise HTTPException(
            status_code=422,
            detail="Couldn't find a clear topic in the conversation yet -- chat a bit more about what you're studying, then try /flashcards again.",
        )

    return FlashcardDeck(title=topic, source="conversation", cards=[Flashcard(**c) for c in cards])


# --- Quiz mode + retry-exam --------------------------------------------
# Same /command pattern as flashcards above: bare "/quiz" and "/retry" are
# intercepted client-side and never reach POST /chat. Grading is explicit
# and client-side too -- correct_index ships down with the quiz, and the
# frontend checks each answer the instant the student picks one -- these
# endpoints only ever generate questions and (via /quiz/results) record
# what actually happened, so /retry-quiz has real misses to work from
# instead of trying to infer them from free-form chat. See agent.py's
# generate_quiz_from_conversation() / record_quiz_results() /
# generate_retry_quiz().


@app.post("/quiz", response_model=QuizDeck)
def generate_quiz_endpoint(req: QuizRequest, request: Request):
    transcript = agent.recent_transcript(req.session_id)
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Nothing to build a quiz from yet -- chat about a topic first, then try /quiz again.",
        )

    _enforce_rate_limit(request, "quiz requests")

    try:
        topic, questions = agent.generate_quiz_from_conversation(transcript, count=5)
    except LLMBusyError:
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
    except Exception as e:
        detail = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Couldn't generate a quiz: {e}"
        raise HTTPException(status_code=502, detail=detail)

    if not questions or not topic:
        raise HTTPException(
            status_code=422,
            detail="Couldn't find a clear topic in the conversation yet -- chat a bit more about what you're studying, then try /quiz again.",
        )

    return QuizDeck(title=topic, source="conversation", questions=[QuizQuestion(**q) for q in questions])


@app.post("/quiz/results")
def record_quiz_results_endpoint(req: QuizResultsRequest):
    if not req.answers:
        raise HTTPException(status_code=422, detail="No answers to record")
    wrong = agent.record_quiz_results(req.session_id, req.topic, [a.model_dump() for a in req.answers])
    return {"status": "recorded", "graded": len(req.answers), "wrong": wrong}


@app.post("/retry-quiz", response_model=QuizDeck)
def generate_retry_quiz_endpoint(req: QuizRequest, request: Request):
    _enforce_rate_limit(request, "quiz requests")

    try:
        result = agent.generate_retry_quiz(req.session_id, count=5)
    except LLMBusyError:
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
    except Exception as e:
        detail = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Couldn't generate a retry quiz: {e}"
        raise HTTPException(status_code=502, detail=detail)

    if result is None:
        detail = (
            "Nice \u2014 you've got everything right so far! Take another /quiz to keep practicing."
            if agent.has_quiz_attempts(req.session_id)
            else "You haven't taken a quiz in this conversation yet -- try /quiz first, then /retry to focus on anything you missed."
        )
        raise HTTPException(status_code=422, detail=detail)

    topic, questions = result
    if not questions:
        raise HTTPException(status_code=502, detail="Couldn't generate a retry quiz -- please try again.")

    return QuizDeck(title=topic, source="retry", questions=[QuizQuestion(**q) for q in questions])


def _prefixed_input(req: ChatRequest) -> str:
    user_input = req.message
    if req.course_code:
        user_input = f"[Course context: {req.course_code}] {user_input}"
    return user_input


NOTEBOOK_RESTRICTION_PROMPT = (
    "The student has pinned specific notebook PDF(s) for this conversation -- "
    "search_notebook is restricted to exactly those document(s), and search_syllabus "
    "isn't available this turn. Treat search_notebook as your only source of grounding: "
    "call it for anything that needs grounding, and if it doesn't cover the question, "
    "say so plainly rather than answering from outside knowledge or general syllabus "
    "familiarity."
)


def _notebook_scope(req: ChatRequest):
    """Resolves the request-scoped search_notebook tool plus, when the
    student has pinned specific PDF(s) via ChatRequest.notebook_doc_ids,
    the extra pieces needed to make "these PDFs only" actually mean only:
    exclude_tool_names leaves search_syllabus (and the syllabus-driven
    build_ai_study_plan) out of the toolset entirely for this call rather
    than just asking the model not to use them, and extra_system_prompt
    tells the model why and what to do instead. Returns
    (extra_tools, exclude_tool_names, extra_system_prompt) -- the last two
    are None/None when nothing is pinned, so behavior for every existing
    caller (no notebook_doc_ids) is unchanged.
    """
    if not req.device_id:
        return None, None, None

    doc_ids = [d.strip() for d in (req.notebook_doc_ids or []) if d.strip()] or None
    extra_tools = [make_notebook_tool(vectorstore, req.device_id, doc_ids=doc_ids)]
    if not doc_ids:
        return extra_tools, None, None

    exclude_tool_names = {"search_syllabus", "build_ai_study_plan"}
    return extra_tools, exclude_tool_names, NOTEBOOK_RESTRICTION_PROMPT


# This is where the frontend sends the users messages
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")
    if len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=422, detail=f"Message is too long (max {MAX_MESSAGE_LENGTH} characters).")

    _enforce_rate_limit(request, "messages")

    extra_tools, exclude_tool_names, extra_system_prompt = _notebook_scope(req)
    result = agent.chat(
        session_id=req.session_id,
        user_input=_prefixed_input(req),
        mode=req.mode,
        extra_tools=extra_tools,
        exclude_tool_names=exclude_tool_names,
        extra_system_prompt=extra_system_prompt,
    )
    return ChatResponse(
        session_id=req.session_id,
        response=result["content"],
        grounded=bool(result["sources"]),
        sources=[SourceChunk(**s) for s in result["sources"]],
        tools_used=result["tools_used"],
        mode=result.get("mode"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """Streaming twin of POST /chat, as Server-Sent Events. Each event is a
    line `data: <json>\\n\\n` with one of these shapes:

      {"type": "tool", "name": "search_syllabus"}
      {"type": "token", "text": "..."}
      {"type": "done", "content", "grounded", "sources", "tools_used", "mode"}
      {"type": "error", "error": "..."}

    Tool orchestration is identical to /chat -- the same guardrails, same
    tools, same grounding behavior. Cancellation is handled by the client
    simply aborting the fetch/EventSource; if the connection drops mid-turn,
    the generator raises GeneratorExit on its next yield and this function
    lets that propagate to stop cleanly rather than trying to keep writing
    to a closed connection.

    The rate-limit check happens here, before the StreamingResponse is
    even constructed, so a throttled request comes back as a normal JSON
    429 the frontend already knows how to parse -- not a malformed or
    empty SSE stream.
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")
    if len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=422, detail=f"Message is too long (max {MAX_MESSAGE_LENGTH} characters).")

    _enforce_rate_limit(request, "messages")

    user_input = _prefixed_input(req)
    extra_tools, exclude_tool_names, extra_system_prompt = _notebook_scope(req)

    def event_stream():
        try:
            for event in agent.chat_stream(
                session_id=req.session_id,
                user_input=user_input,
                mode=req.mode,
                extra_tools=extra_tools,
                exclude_tool_names=exclude_tool_names,
                extra_system_prompt=extra_system_prompt,
            ):
                if event.get("type") == "done":
                    event = {**event, "grounded": bool(event.get("sources"))}
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Resets the session, optionally returning a short recap first
@app.delete("/chat/{session_id}", response_model=ResetResponse)
def reset_session(session_id: str, request: Request, summarize: bool = Query(False)):
    if summarize:
        _enforce_rate_limit(request, "summary requests")
    summary = agent.summarize(session_id) if summarize else None
    agent.reset(session_id)
    return ResetResponse(status="reset", session_id=session_id, summary=summary)


# --- Same-origin frontend ----------------------------------------------
# Must be the LAST route added: FastAPI/Starlette checks routes in the
# order they were registered, and a Mount("/") matches any path as a
# prefix -- so every API route above still gets first look at a request,
# and this mount only ever ends up serving an actual static file (or,
# with html=True, index.html for "/" and any other directory-style path).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
