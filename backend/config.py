"""
Centralized environment-driven configuration.

Everything here has a default that reproduces pre-v1.4 behavior when its
environment variable is unset, so an existing local setup (the README's
`export GROQ_API_KEY=...` and nothing else) keeps working exactly as
before with zero changes. Deploying the public demo link then becomes a
matter of setting a handful of these in `.env` -- see `.env.example` for
what each one does and what a sensible demo value looks like.

This module intentionally stays dependency-free (stdlib `os` only, same
as the rest of the backend) rather than pulling in pydantic-settings for
what's currently a dozen scalars.

Settings land here incrementally as each piece of the production-
readiness round (v1.4) is built -- this file starts with just the model
name and the demo-mode/CORS basics; later patches in the same round add
to it rather than introducing a second config module.
"""
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Model ------------------------------------------------------------------
# Previously hardcoded in main.py. Overriding this is mainly useful for a
# cheaper/faster fallback during local testing; the production demo should
# leave it on the default.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"

# --- Demo-mode protections ---------------------------------------------------
# DEMO_MODE doesn't gate anything by itself -- each guarded feature (shared-
# course write protection, visitor-data TTL, etc.) checks it individually as
# those land later in this same round, so turning it on is additive rather
# than one all-or-nothing switch. Off by default so local development is
# completely unaffected by this patch.
DEMO_MODE = _env_bool("DEMO_MODE", False)

# A hidden token that unlocks shared-course add/delete while DEMO_MODE is on
# (enforced starting in the "shared-course write protection" patch later in
# this round -- defined here now so one .env holds every v1.4 setting).
# Unset means "no admin access", not "wide open": DEMO_MODE=true with no
# token configured simply locks shared-course writes entirely rather than
# failing open.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip() or None

# --- CORS ---------------------------------------------------------------
# Comma-separated origins, e.g. "https://my-tunnel.trycloudflare.com". Falls
# back to "*" (today's behavior) when unset, which is fine for local dev
# but should be set to the real tunnel/deploy origin before sharing a public
# link. Once the same-origin-frontend patch lands, the browser won't need
# CORS at all for the shipped frontend -- this only matters for a request
# made from a *different* origin (a second tool, a manual fetch, etc.).
ALLOWED_ORIGINS = _env_list("ALLOWED_ORIGINS", ["*"])

# --- Rate limiting ---------------------------------------------------------
# See backend/rate_limit.py. RATE_LIMIT_PER_MINUTE is the existing default,
# now configurable; RATE_LIMIT_PER_DAY is new -- a per-visitor ceiling for a
# long demo day, well above what a real student session would ever need,
# meant to catch a bot or a forgotten open tab rather than a person.
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 15)
RATE_LIMIT_PER_DAY = _env_int("RATE_LIMIT_PER_DAY", 300)

# --- Input / resource caps --------------------------------------------------
# A generous ceiling for one chat message -- comfortably more than a real
# question needs, but bounds how much a single turn can grow the prompt
# sent to Groq.
MAX_MESSAGE_LENGTH = _env_int("MAX_MESSAGE_LENGTH", 4000)

# How many of a session's most recent turns (one turn = one student message
# through the assistant's reply, tool calls included) stay in the prompt
# sent to the model. Bounds a long-running session's token usage and
# latency instead of resending the entire conversation forever -- see
# agent.py's _history(), which is the only place this is read. A "turn" is
# never split mid tool-call/tool-response pair; see that method's docstring
# for why that matters.
HISTORY_WINDOW_TURNS = _env_int("HISTORY_WINDOW_TURNS", 20)

# Per-device cap on the private notebook (see notebook_store.py) -- a demo
# visitor's browser, not a real account, so these exist to keep one device
# from growing the shared notebook_docs.json file without bound rather than
# to model a real product quota.
NOTEBOOK_MAX_DOCS = _env_int("NOTEBOOK_MAX_DOCS", 5)
NOTEBOOK_MAX_MB = _env_int("NOTEBOOK_MAX_MB", 50)

# --- Data lifecycle ---------------------------------------------------------
# Visitor data -- chat sessions (and the quiz history tied to them) and
# notebook uploads -- is swept away after this many hours of inactivity,
# by a background task started in main.py (see its lifespan handler), not
# a real cron job, since this is a single demo process. Not gated behind
# DEMO_MODE: an idle local-dev session is presumably abandoned too, so
# there's no reason to only bound memory/disk growth in demo mode.
DATA_TTL_HOURS = _env_int("DATA_TTL_HOURS", 24)

# --- Resilience --------------------------------------------------------
# Bounds how long the backend waits on a single Groq HTTP request (set on
# the ChatGroq client itself, see main.py) -- a stalled connection can't
# hang a worker thread forever.
LLM_CALL_TIMEOUT_SECONDS = _env_int("LLM_CALL_TIMEOUT_SECONDS", 30)

# How many Groq calls -- chat, flashcards, quiz, all of them -- are
# allowed to run at once, across every visitor combined (see
# agent.py's _invoke_with_backoff / _stream_with_backoff, the two choke
# points every Groq call in this codebase actually goes through). Past
# this, a new call waits up to LLM_QUEUE_WAIT_SECONDS for a free slot
# before giving up with a friendly "busy" message instead of piling up
# indefinitely -- degrading politely under load rather than falling over.
MAX_CONCURRENT_LLM_CALLS = _env_int("MAX_CONCURRENT_LLM_CALLS", 5)
LLM_QUEUE_WAIT_SECONDS = _env_int("LLM_QUEUE_WAIT_SECONDS", 10)

# --- Logging -------------------------------------------------------------
# Plain text to stdout (see main.py's logging.basicConfig call) -- no file
# rotation, no JSON formatter; this is a single demo process, not a fleet.
LOG_LEVEL = (os.environ.get("LOG_LEVEL", "INFO").strip().upper()) or "INFO"
