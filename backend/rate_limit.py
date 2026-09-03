"""
Lightweight per-visitor request throttle.

Not about cost -- Groq's free tier is free -- this exists so a single
heavy tester hitting the public demo link can't burn through the
shared per-key rate limit Groq enforces and lock everyone else out.
"A broken public demo is worse for a CV than one that degrades
politely" (see roadmap item 3).

Keyed by client IP address for now. The original plan was to key this
by `device_id` (a persistent id stored in the browser's localStorage),
but that id belongs to the PDF-notebook-import work and doesn't exist
yet. IP is available today with zero frontend changes and is "good
enough" for the resilience goal here -- swap the key over to device_id
later if per-browser-tab granularity ends up mattering more than
per-network granularity.

Fixed-window counter, in-memory, single process. Good enough for a
demo server; would need a shared store (e.g. Redis) behind multiple
uvicorn workers or a horizontally-scaled deploy.
"""
import time
from collections import defaultdict
from threading import Lock

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 15

_lock = Lock()
# key -> (window_start_monotonic, count_in_window)
_windows: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))


def check_and_increment(key: str) -> tuple[bool, int]:
    """Records one request for `key` and reports whether it's allowed.

    Returns (allowed, retry_after_seconds). retry_after_seconds is 0
    when allowed is True, otherwise the number of seconds until the
    current window rolls over and the caller can try again.
    """
    now = time.monotonic()
    with _lock:
        window_start, count = _windows[key]
        elapsed = now - window_start

        if window_start == 0.0 or elapsed >= WINDOW_SECONDS:
            _windows[key] = (now, 1)
            return True, 0

        if count >= MAX_REQUESTS_PER_WINDOW:
            return False, int(WINDOW_SECONDS - elapsed) + 1

        _windows[key] = (window_start, count + 1)
        return True, 0
