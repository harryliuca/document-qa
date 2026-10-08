import asyncio
import io
import json
import re

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.config import Settings
from app.main import create_app
from app.models import Evidence, ModelAnswer


class FakeProvider:
    """Deterministic provider: network-free endpoint tests, never used by the shipped app."""

    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []
        self.embed_calls = 0
        self.active = 0
        self.peak = 0

    async def embed(self, texts):
        self.embed_calls += 1
        vocabulary = ("aws", "incident", "backups", "revenue", "oregon", "notification")
        return [[float(word in text.lower()) + 0.01 for word in vocabulary] for text in texts], 123

    async def answer(self, question, chunks):
        self.calls.append(question)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if "revenue" in question.lower():
                return ModelAnswer(status="not_found", answer="", evidence=[]), 30, 10
            terms = re.findall(r"\w+", question.lower())
            chunk = max(chunks, key=lambda c: sum(t in c.text.lower() for t in terms))
            return (
                ModelAnswer(
                    status="answered",
                    answer="Supported answer: " + chunk.text,
                    evidence=[Evidence(chunk_id=chunk.id, quote=chunk.text)],
                ),
                50,
                20,
            )
        finally:
            self.active -= 1


@pytest.fixture
def settings():
    return Settings(_env_file=None, openai_api_key="", parser_timeout=30)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(settings, provider):
    with TestClient(create_app(settings, provider)) as test_client:
        yield test_client


def pdf_bytes(text="Production runs on AWS in Oregon.", pages=1):
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    for _ in range(pages):
        if text:
            pdf.drawString(50, 760, text)
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def upload(document=None, questions=None, name="policy.json"):
    return {
        "document": (name, document if document is not None else b'{"hosting":"AWS in Oregon"}'),
        "questions": ("questions.json", json.dumps(questions or ["Where is hosting?"]).encode()),
    }
