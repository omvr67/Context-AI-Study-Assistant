"""
Lightweight per-visitor request throttle.

Not about cost -- Groq's free tier is free -- this exists so a single
heavy tester (or a bot) hitting the public demo link can't burn through
the shared per-key rate limit Groq enforces and lock everyone else out,
and can't quietly exhaust a whole day's quota before anyone else gets to
try the demo. "A broken public demo is worse for a CV than one that
degrades politely" (see roadmap item 3).

Two independent fixed windows, both from backend/config.py:
RATE_LIMIT_PER_MINUTE absorbs ordinary bursts (someone mashing a button);
RATE_LIMIT_PER_DAY is the quota-protection ceiling for a long demo day.
A request must fit under *every* configured window to be allowed -- see
check_and_increment's docstring for exactly how a request that's blocked
by one window interacts with the others.

Keyed by client IP address (proxy-aware -- see main.py's _client_key) for
now. The original plan was to key this by `device_id` (a persistent id
stored in the browser's localStorage), but that id belongs to the
PDF-notebook-import work and doesn't exist yet. IP is available today
with zero frontend changes and is "good enough" for the resilience goal
here -- swap the key over to device_id later if per-browser-tab
granularity ends up mattering more than per-network granularity.

Fixed-window counters, in-memory, single process. Good enough for a demo
server; would need a shared store (e.g. Redis) behind multiple uvicorn
workers or a horizontally-scaled deploy.
"""
import time
from collections import defaultdict
from threading import Lock

from .config import RATE_LIMIT_PER_DAY, RATE_LIMIT_PER_MINUTE

# name -> (window_seconds, max_requests_in_window)
WINDOWS: dict[str, tuple[int, int]] = {
    "minute": (60, RATE_LIMIT_PER_MINUTE),
    "day": (86400, RATE_LIMIT_PER_DAY),
}

_lock = Lock()
# (window_name, key) -> (window_start_monotonic, count_in_window)
_counters: dict[tuple[str, str], tuple[float, int]] = defaultdict(lambda: (0.0, 0))


def check_and_increment(key: str) -> tuple[bool, int]:
    """Records one request for `key` against every configured window and
    reports whether it's allowed under all of them.

    Returns (allowed, retry_after_seconds). retry_after_seconds is 0 when
    allowed is True. When allowed is False, it's the real number of
    seconds until whichever blocking window clears -- if the daily cap is
    what's actually blocking, this reports the true (long) wait rather
    than a shorter, more optimistic number borrowed from the per-minute
    window, so the caller can give an honest message instead of inviting
    an immediate retry that will just fail again.

    A request that's blocked by any window doesn't consume quota in any
    window -- windows are checked (not mutated) first, and only applied
    together once every window has room, so a rejected request costs the
    visitor nothing.
    """
    now = time.monotonic()
    with _lock:
        pending: dict[tuple[str, str], tuple[float, int]] = {}
        blocked_retry_after = 0

        for name, (seconds, limit) in WINDOWS.items():
            window_start, count = _counters[(name, key)]
            elapsed = now - window_start

            if window_start == 0.0 or elapsed >= seconds:
                pending[(name, key)] = (now, 1)
                continue

            if count >= limit:
                blocked_retry_after = max(blocked_retry_after, int(seconds - elapsed) + 1)
                continue

            pending[(name, key)] = (window_start, count + 1)

        if blocked_retry_after:
            return False, blocked_retry_after

        _counters.update(pending)
        return True, 0
