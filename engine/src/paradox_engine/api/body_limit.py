from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Bound the body before JSON decoding, including chunked requests."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    length = int(value)
                except ValueError:
                    length = -1
                if length < 0:
                    await JSONResponse(
                        {"detail": "Invalid Content-Length"}, status_code=400
                    )(scope, receive, send)
                    return
                if length > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"detail": "Request body too large"}, status_code=413)(
            scope, receive, send
        )
