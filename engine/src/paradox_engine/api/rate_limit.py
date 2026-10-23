import asyncio
import math
import time
from collections.abc import Callable

from fastapi import HTTPException, Request


class IPRateLimiter:
    """Bounded, process-local minimum-interval limiter keyed by client IP."""

    def __init__(
        self, clock: Callable[[], float] = time.monotonic, *, max_entries: int = 10_000
    ) -> None:
        self.clock = clock
        self.max_entries = max_entries
        self._expires_at: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()
        self._request_count = 0

    async def check(self, *, scope: str, client_ip: str, interval: float) -> None:
        key = (scope, client_ip)
        async with self._lock:
            now = self.clock()
            expires_at = self._expires_at.get(key)
            if expires_at is not None:
                retry_after = expires_at - now
                if retry_after > 0:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": str(math.ceil(retry_after))},
                    )
            self._request_count += 1
            if (
                self._request_count % 256 == 0
                or len(self._expires_at) >= self.max_entries
            ):
                self._expires_at = {
                    stored_key: expiry
                    for stored_key, expiry in self._expires_at.items()
                    if expiry > now
                }
            if (
                key not in self._expires_at
                and len(self._expires_at) >= self.max_entries
            ):
                raise HTTPException(
                    status_code=503,
                    detail="Rate limiter capacity exceeded",
                    headers={"Retry-After": "60"},
                )
            self._expires_at[key] = now + interval


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


async def limit_generation(request: Request) -> None:
    await request.app.state.rate_limiter.check(
        scope="generation",
        client_ip=_client_ip(request),
        interval=15,
    )
