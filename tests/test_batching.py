import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.batching import FullContextService, plan_batches
from app.full_context_provider import batch_messages, request_tokens, serialize_sources
from app.ingestion import strict_json
from app.main import create_app
from app.models import (
    BatchItem,
    BatchOutput,
    Chunk,
    ClientFault,
    Evidence,
    ProviderFault,
    ReviewItem,
    ReviewOutput,
    Usage,
)
from tests.conftest import upload

CHUNKS = [
    Chunk("c1", "Backups are encrypted. Backups are retained for 30 days.", "$.security.backups")
]


class BatchFake:
    def __init__(self, mode="supported"):
        self.mode, self.calls, self.reviews = mode, [], []
        self.active = self.peak = 0

    async def generate_batch(self, source, questions, focus, max_output_tokens):
        self.calls.append((source, questions, focus))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        if self.mode == "outage":
            raise ProviderFault("provider_limit", "Unavailable")
        if self.mode == "timeout":
            raise TimeoutError
        records = json.loads(source)["untrusted_document"]
        evidence = [Evidence(chunk_id=records[0]["chunk_id"], quote=records[0]["text"])]
        items = [
            BatchItem(
                question_id=q["question_id"], status="answered", answer="30 days", evidence=evidence
            )
            for q in questions
        ]
        if self.mode == "missing" and not focus:
            items = items[1:]
        if self.mode == "duplicate" and not focus:
            items.append(items[0])
        if self.mode == "unknown" and not focus:
            items.append(items[0].model_copy(update={"question_id": "unknown"}))
        if self.mode == "bad_quote":
            items[0].evidence = [Evidence(chunk_id="c1", quote="invented")]
        if self.mode == "not_found":
            for item in items:
                item.status, item.answer, item.evidence = "not_found", "", []
        if self.mode == "ambiguous":
            items[0].status = "needs_review"
        return BatchOutput(results=list(reversed(items))), Usage(
            input_tokens=100, cached_input_tokens=64, output_tokens=20
        )

    async def review_batch(self, candidates):
        self.reviews.append(candidates)
        if self.mode == "review_outage":
            raise ProviderFault("provider_unavailable", "Unavailable")
        verdict = (
            self.mode
            if self.mode in {"contradicted", "incomplete", "insufficient_evidence"}
            else "supported"
        )
        return ReviewOutput(
            results=[ReviewItem(question_id=c["question_id"], verdict=verdict) for c in candidates]
        ), Usage(input_tokens=50, output_tokens=10)


def service(settings, fake):
    return FullContextService(fake, settings, asyncio.Semaphore(2))


def test_topic_plan_deduplicates_and_preserves_ids():
    unique, batches = plan_batches(
        ["Backup retention?", "IAM permissions?", "Backup retention?", "Encryption?"], 2
    )
    assert [q["question"] for q in unique] == [
        "Backup retention?",
        "IAM permissions?",
        "Encryption?",
    ]
    assert [q["question_id"] for q in batches[0]] == ["q0002", "q0003"]


async def test_supported_order_duplicate_and_usage(settings):
    fake = BatchFake()
    answers, usage = await service(settings, fake).run(CHUNKS, ["Backup?", "IAM?", "Backup?"])
    assert [a.question for a in answers] == ["Backup?", "IAM?", "Backup?"]
    assert all(a.evidence_check == "model_checked" for a in answers)
    assert usage.llm_calls == 2 and usage.verification_calls == 1
    assert usage.cached_input_tokens == 64 and usage.input_tokens == 150
    assert len(fake.calls[0][1]) == 2
    assert fake.reviews[0][0]["sources"][0]["location"] == "$.security.backups"


@pytest.mark.parametrize("mode,count", [("missing", 1), ("duplicate", 1), ("unknown", 2)])
async def test_malformed_ids_retry_only_affected_once(settings, mode, count):
    fake = BatchFake(mode)
    answers, usage = await service(settings, fake).run(CHUNKS, ["Backup?", "IAM?"])
    assert all(a.status == "answered" for a in answers)
    assert usage.retried_questions == count
    assert len(fake.calls) == 1 + count
    assert all(
        len(call[1]) == 1 and call[2]["focus_chunk_ids"] == ["c1"] for call in fake.calls[1:]
    )
    assert len({call[0] for call in fake.calls}) == 1


@pytest.mark.parametrize(
    "mode", ["bad_quote", "ambiguous", "contradicted", "incomplete", "insufficient_evidence"]
)
async def test_unverified_drafts_withheld_after_single_retry(settings, mode):
    fake = BatchFake(mode)
    answers, usage = await service(settings, fake).run(CHUNKS, ["Backup?"])
    assert answers[0].status == "needs_review" and answers[0].attempts == 2
    assert answers[0].citations == [] and "30 days" not in answers[0].answer
    assert len(fake.calls) == 2 and usage.retried_questions == 1


@pytest.mark.parametrize(
    "mode,status",
    [
        ("not_found", "not_found"),
        ("outage", "error"),
        ("timeout", "error"),
        ("review_outage", "needs_review"),
    ],
)
async def test_abstention_and_outages_do_not_repeat_spend(settings, mode, status):
    fake = BatchFake(mode)
    answers, usage = await service(settings, fake).run(CHUNKS, ["Backup?"])
    assert answers[0].status == status and len(fake.calls) == 1
    assert usage.retried_questions == 0
    assert usage.usage_complete == (mode == "not_found")


async def test_preflight_budget_before_any_provider_calls(settings):
    fake = BatchFake()
    settings.full_context_input_tokens = 1000
    with pytest.raises(ClientFault, match="full_context_limit"):
        await service(settings, fake).run(CHUNKS, ["Backup?"])
    assert fake.calls == []


async def test_warm_first_and_concurrency(settings):
    settings.full_context_batch_size = 1
    fake = BatchFake()
    answers, _ = await service(settings, fake).run(
        CHUNKS, ["IAM?", "Encryption?", "Backup?", "Revenue?"], "warm_first"
    )
    assert fake.calls[0][1][0]["question"] == "IAM?"
    assert len(answers) == 4 and fake.peak <= 2
    assert len(fake.reviews) == 4


def test_prefix_and_schema_in_token_budget():
    source = serialize_sources(CHUNKS)
    one = batch_messages(source, [{"question_id": "q1", "question": "A?"}], {})
    two = batch_messages(source, [{"question_id": "q2", "question": "B?"}], {"reason": "ambiguous"})
    assert one[:2] == two[:2]
    assert one[1]["role"] == "user"
    assert request_tokens(one, BatchOutput) > 1024


def test_full_endpoint_and_limit(settings, provider):
    fake = BatchFake()
    with TestClient(create_app(settings, provider, fake)) as client:
        response = client.post(
            "/api/answers?strategy=full_context_batch&cache_mode=warm_first",
            files=upload(questions=["Backup?"] * 75),
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["strategy"] == "full_context_batch" and result["cache_mode"] == "warm_first"
        assert len(result["results"]) == 75 and provider.embed_calls == 0
        assert (
            client.post(
                "/api/answers?strategy=full_context_batch", files=upload(questions=["Q?"] * 76)
            ).status_code
            == 422
        )
        assert client.post("/api/answers?strategy=invalid", files=upload()).status_code == 422


@pytest.mark.parametrize("raw", [b'{"x":1e999}', b'{"x":-1e999}'])
def test_overflow_json_numbers_rejected(raw):
    with pytest.raises(ClientFault):
        strict_json(raw)
