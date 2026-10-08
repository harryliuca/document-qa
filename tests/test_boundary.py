import asyncio
import json

import pytest

from app.middleware import RequestBoundary


async def invoke(boundary, body=b"", receive_override=None):
    sent = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await boundary(
        {"type": "http", "path": "/api/answers", "method": "POST", "headers": []},
        receive_override or receive,
        send,
    )
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(data)


async def test_overload_is_rejected_and_capacity_recovers(settings):
    started, release = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        started.set()
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    settings.max_concurrent_requests = 1
    boundary = RequestBoundary(app, settings)
    first = asyncio.create_task(invoke(boundary))
    await started.wait()
    status, data = await invoke(boundary)
    assert status == 429 and data["error"]["code"] == "busy"
    release.set()
    assert (await first)[0] == 200
    assert boundary.active == 0


@pytest.mark.parametrize("phase,expected", [("upload", 408), ("request", 504)])
async def test_timeouts_release_capacity(settings, phase, expected):
    settings.upload_timeout = 0.005
    settings.request_timeout = 0.005

    async def slow_receive():
        await asyncio.sleep(1)

    async def app(scope, receive, send):
        await asyncio.sleep(1)

    boundary = RequestBoundary(app, settings)
    status, _ = await invoke(boundary, receive_override=slow_receive if phase == "upload" else None)
    assert status == expected
    assert boundary.active == 0


async def test_unexpected_errors_are_sanitized(settings):
    async def app(scope, receive, send):
        raise ValueError("secret content")

    status, data = await invoke(RequestBoundary(app, settings))
    assert status == 500
    assert "secret" not in json.dumps(data)
