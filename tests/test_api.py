import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import ProviderFault
from tests.conftest import FakeProvider, pdf_bytes, upload


def test_json_endpoint_and_metadata(client, provider):
    response = client.post(
        "/api/answers", files=upload(questions=["Hosting?", "Revenue?", "Hosting?"])
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [r["status"] for r in body["results"]] == ["answered", "not_found", "answered"]
    assert len(provider.calls) == 2
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["results"][0]["citations"][0]["location"].startswith("JSON pointer")
    assert body["usage"]["embedding_tokens"] == 123


def test_pdf_endpoint(client):
    response = client.post("/api/answers", files=upload(pdf_bytes(), name="policy.pdf"))
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["citations"][0]["location"] == "page 1"


@pytest.mark.parametrize(
    "document,name,status,code",
    [
        (b"bad", "doc.pdf", 422, "invalid_pdf"),
        (b"{}", "doc.json", 422, "empty_document"),
        (b"bad", "doc.json", 422, "invalid_json"),
        (b"x", "doc.exe", 415, "unsupported_type"),
    ],
)
def test_document_errors(client, document, name, status, code):
    response = client.post("/api/answers", files=upload(document, name=name))
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_malformed_questions_and_missing_files(client, provider):
    files = upload()
    files["questions"] = ("q.json", b"broken")
    assert client.post("/api/answers", files=files).status_code == 422
    assert client.post("/api/answers", files={"document": ("x.json", b"{}")}).status_code == 422
    assert provider.embed_calls == 0


def test_extra_fields_and_cross_site_rejected(client):
    assert client.post("/api/answers", files=upload(), data={"extra": "bad"}).status_code == 400
    assert (
        client.post(
            "/api/answers", files=upload(), headers={"Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 403
    )


def test_large_body_rejected_before_parsing(settings):
    settings.max_body_bytes = 1024
    with TestClient(create_app(settings, FakeProvider())) as client:
        response = client.post("/api/answers", content=b"x" * 1025)
    assert response.status_code == 413


def test_access_token(settings):
    from pydantic import SecretStr

    settings.app_api_token = SecretStr("local-test-token")
    with TestClient(create_app(settings, FakeProvider())) as client:
        assert client.post("/api/answers", files=upload()).status_code == 401
        assert (
            client.post(
                "/api/answers", files=upload(), headers={"Authorization": "Bearer local-test-token"}
            ).status_code
            == 200
        )


def test_no_key_still_serves_ui_and_health(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok", "ai_configured": False}
        assert client.post("/api/answers", files=upload()).status_code == 503


def test_provider_failure_sanitized(settings):
    class FailedProvider(FakeProvider):
        async def embed(self, texts):
            raise ProviderFault("provider_limit", "AI quota or rate limit reached. Try later.")

    with TestClient(create_app(settings, FailedProvider())) as client:
        response = client.post("/api/answers", files=upload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_limit"


def test_logs_do_not_include_document_or_questions(client, caplog):
    private = "CONFIDENTIAL-CUSTOMER-123"
    caplog.set_level(logging.INFO, logger="document_qa")
    response = client.post(
        "/api/answers", files=upload(json.dumps({"secret": private}).encode(), [private + "?"])
    )
    assert response.status_code == 200
    records = [json.loads(r.message) for r in caplog.records if r.name == "document_qa"]
    assert any(r["event"] == "qa_complete" for r in records)
    assert private not in json.dumps(records)


def test_static_security_headers_and_openapi(client):
    response = client.get("/")
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    schema = client.get("/openapi.json").json()
    assert (
        "multipart/form-data" in schema["paths"]["/api/answers"]["post"]["requestBody"]["content"]
    )
