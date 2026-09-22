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
