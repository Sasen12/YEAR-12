"""In-memory rate limiter for lightweight endpoint protection."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Simple fixed-window limiter per key."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        now = time.monotonic()
        retry_after = 0
        with self._lock:
            q = self._hits[key]
            cutoff = now - window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_requests:
                retry_after = max(1, int(window_seconds - (now - q[0])))
                return False, retry_after
            q.append(now)
        return True, retry_after
