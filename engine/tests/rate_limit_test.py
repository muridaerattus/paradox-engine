import asyncio

import pytest
from fastapi import HTTPException

from paradox_engine.api.rate_limit import IPRateLimiter


def test_rate_limits_are_scoped_and_expire():
    now = [100.0]
    limiter = IPRateLimiter(clock=lambda: now[0])

    asyncio.run(limiter.check(scope="threads", client_ip="192.0.2.1", interval=60))
    asyncio.run(limiter.check(scope="messages", client_ip="192.0.2.1", interval=3))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(limiter.check(scope="threads", client_ip="192.0.2.1", interval=60))
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "60"}

    now[0] += 60
    asyncio.run(limiter.check(scope="threads", client_ip="192.0.2.1", interval=60))


def test_rate_limits_are_independent_per_ip():
    limiter = IPRateLimiter(clock=lambda: 100.0)
    asyncio.run(limiter.check(scope="threads", client_ip="192.0.2.1", interval=60))
    asyncio.run(limiter.check(scope="threads", client_ip="192.0.2.2", interval=60))


def test_capacity_is_bounded_without_evicting_active_limits():
    now = [100.0]
    limiter = IPRateLimiter(clock=lambda: now[0], max_entries=1)
    asyncio.run(limiter.check(scope="generation", client_ip="192.0.2.1", interval=15))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            limiter.check(scope="generation", client_ip="192.0.2.2", interval=15)
        )
    assert exc_info.value.status_code == 503
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            limiter.check(scope="generation", client_ip="192.0.2.1", interval=15)
        )
    assert exc_info.value.status_code == 429
    now[0] += 15
    asyncio.run(limiter.check(scope="generation", client_ip="192.0.2.2", interval=15))


def test_expiration_respects_intervals_longer_than_sixty_seconds():
    now = [100.0]
    limiter = IPRateLimiter(clock=lambda: now[0], max_entries=1)
    asyncio.run(limiter.check(scope="long", client_ip="192.0.2.1", interval=120))
    now[0] += 61
    with pytest.raises(HTTPException):
        asyncio.run(limiter.check(scope="other", client_ip="192.0.2.2", interval=3))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(limiter.check(scope="long", client_ip="192.0.2.1", interval=120))
    assert exc_info.value.headers == {"Retry-After": "59"}
