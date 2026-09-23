from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

LIMITED_ROUTES = frozenset(
    {
        ("POST", "/v1/ingest"),
        ("POST", "/v1/ingest/image"),
        ("GET", "/v1/search"),
        ("POST", "/v1/ask"),
        ("POST", "/v1/agent"),
        ("POST", "/v1/agent/stream"),
        ("POST", "/v1/agent/resume"),
    }
)


class RateLimiter:
    """Fixed window of hits per key. A max of 0 allows every call."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self._max_requests <= 0:
            return True
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self._max_requests:
                return False
            hits.append(now)
            return True
