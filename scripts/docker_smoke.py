"""Runtime checks against the Compose app. AI responses come from the test-only mock server."""

import io
import json
from pathlib import Path

import httpx
from pypdf import PdfReader, PdfWriter


def main():
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=45) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["ai_configured"]
        home = client.get("/")
        assert home.status_code == 200 and "full_context_batch" in home.text
        assert client.get("/static/app.js").status_code == 200
        print("PASS: health, HTML, and JavaScript")
        for filename, strategy in [
            ("policy.json", "retrieval"),
            ("policy.pdf", "full_context_batch"),
        ]:
            response = client.post(
                "/api/answers",
                params={"strategy": strategy},
                files={
                    "document": (filename, Path("fixtures", filename).read_bytes()),
                    "questions": ("questions.json", json.dumps(["Which cloud provider is used?"])),
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["strategy"] == strategy
            assert result["results"][0]["status"] == "answered"
            assert result["results"][0]["citations"]
            if strategy == "full_context_batch":
                assert result["usage"]["verification_calls"] == 1
                assert result["results"][0]["evidence_check"] == "model_checked"
            print(f"PASS: {filename}, {strategy}, structured answer and evidence")
        writer = PdfWriter()
        writer.append(PdfReader("fixtures/policy.pdf"))
        writer.encrypt("", owner_password="test-owner", algorithm="AES-256")
        buffer = io.BytesIO()
        writer.write(buffer)
        response = client.post(
            "/api/answers?strategy=full_context_batch",
            files={
                "document": ("aes.pdf", buffer.getvalue()),
                "questions": ("questions.json", '["Which cloud provider is used?"]'),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["status"] == "answered"
        print("PASS: AES-encrypted PDF with empty opening password")
        response = client.post(
            "/api/answers",
            files={
                "document": ("bad.pdf", b"%PDF-broken"),
                "questions": ("questions.json", '["Question?"]'),
            },
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "invalid_pdf"
        print("PASS: malformed PDF returns sanitized 422")
    print("Docker runtime smoke test passed. AI is mocked; no real API key or paid calls.")


if __name__ == "__main__":
    main()
