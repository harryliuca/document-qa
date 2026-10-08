"""Bound uploads before multipart parsing; reject overload rather than queue unbounded work."""

import asyncio
import hmac
import json
import logging
import time
import uuid

from starlette.responses import JSONResponse

from app.config import Settings

logger = logging.getLogger("document_qa")


def log_event(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":")))


class RequestBoundary:
    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started, status, response_started = time.monotonic(), 500, False

        async def safe_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                message["headers"] += [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; script-src 'self'; style-src 'self'; "
                        b"frame-ancestors 'none'; base-uri 'none'",
                    ),
                ]
            await send(message)

        async def fail(code: str, message: str, status_code: int):
            await JSONResponse(
                {"error": {"code": code, "message": message}, "request_id": request_id},
                status_code=status_code,
            )(scope, receive, safe_send)

        admitted = False
        try:
            if scope["path"] == "/api/answers" and scope["method"] == "POST":
                headers = dict(scope["headers"])
                token = self.settings.app_api_token.get_secret_value()
                if token and not hmac.compare_digest(
                    headers.get(b"authorization", b""), ("Bearer " + token).encode()
                ):
                    return await fail("unauthorized", "A valid access token is required.", 401)
                # Same-origin browser requests only. CLI clients do not send this header.
                if headers.get(b"sec-fetch-site") == b"cross-site":
                    return await fail("cross_site", "Use the app from its own origin.", 403)
                if self.active >= self.settings.max_concurrent_requests:
                    return await fail("busy", "Server is busy. Please retry shortly.", 429)
                self.active += 1
                admitted = True
                chunks, size = [], 0
                try:
                    async with asyncio.timeout(self.settings.upload_timeout):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            data = message.get("body", b"")
                            size += len(data)
                            if size > self.settings.max_body_bytes:
                                return await fail(
                                    "upload_too_large", "Total upload exceeds 11 MB.", 413
                                )
                            chunks.append(data)
                            if not message.get("more_body", False):
                                break
                except TimeoutError:
                    return await fail("upload_timeout", "Upload took too long. Please retry.", 408)
                body = b"".join(chunks)
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await receive()

                async with asyncio.timeout(self.settings.request_timeout):
                    await self.app(scope, replay, safe_send)
            else:
                await self.app(scope, receive, safe_send)
        except TimeoutError:
            if not response_started:
                await fail(
                    "request_timeout", "Request exceeded its time limit. Try fewer questions.", 504
                )
        except Exception as exc:
            # Never log exception strings or tracebacks: SDK/parser errors can contain input.
            log_event("request_error", request_id=request_id, exception_type=type(exc).__name__)
            if not response_started:
                await fail("internal_error", "Unexpected server error. Please retry.", 500)
        finally:
            if admitted:
                self.active -= 1
            log_event(
                "request_complete",
                request_id=request_id,
                status=status,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
