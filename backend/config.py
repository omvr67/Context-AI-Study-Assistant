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
