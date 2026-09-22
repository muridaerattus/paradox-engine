import asyncio
import math
import time
from collections.abc import Callable

from fastapi import HTTPException, Request


class IPRateLimiter:
    """Process-local sliding-window rate limiter keyed by client IP."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._last_requests: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()
        self._request_count = 0

    async def check(self, *, scope: str, client_ip: str, interval: float) -> None:
        now = self.clock()
        key = (scope, client_ip)
        async with self._lock:
            last_request = self._last_requests.get(key)
            if last_request is not None:
                retry_after = interval - (now - last_request)
                if retry_after > 0:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": str(math.ceil(retry_after))},
                    )
            self._last_requests[key] = now
            self._request_count += 1
            if self._request_count % 256 == 0:
                cutoff = now - 60
                self._last_requests = {
                    stored_key: timestamp
                    for stored_key, timestamp in self._last_requests.items()
                    if timestamp >= cutoff
                }


def _client_ip(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


async def limit_new_threads(request: Request) -> None:
    await request.app.state.rate_limiter.check(
        scope="classpect-thread-create",
        client_ip=_client_ip(request),
        interval=60,
    )


async def limit_thread_messages(request: Request) -> None:
    await request.app.state.rate_limiter.check(
        scope="classpect-thread-message",
        client_ip=_client_ip(request),
        interval=3,
    )
