import asyncio

import pytest

from app.models import Chunk, ClientFault, Evidence, ModelAnswer, ProviderFault
from app.retrieval import HybridIndex
from app.service import QAService, validate_answer
from tests.conftest import FakeProvider

CHUNKS = [Chunk("c0001", "Production is hosted on AWS in Oregon.", "page 1")]


def test_citation_quote_and_id_validation():
    result = ModelAnswer(
        status="answered",
        answer="AWS",
        evidence=[Evidence(chunk_id="c0001", quote="hosted on AWS in Oregon.")],
    )
    answer = validate_answer("Provider?", result, CHUNKS)
    assert answer.citations[0].location == "page 1"
    for evidence in [
        Evidence(chunk_id="wrong", quote=CHUNKS[0].text),
        Evidence(chunk_id="c0001", quote="hosted on Azure"),
        Evidence(chunk_id="c0001", quote=""),
    ]:
        result.evidence = [evidence]
        with pytest.raises(ProviderFault):
            validate_answer("Provider?", result, CHUNKS)
    result.evidence = []
    with pytest.raises(ProviderFault):
        validate_answer("Provider?", result, CHUNKS)


def test_not_found_never_leaks_unverified_model_prose():
    result = ModelAnswer(status="not_found", answer="Imaginary revenue $42M", evidence=[])
    answer = validate_answer("Revenue?", result, CHUNKS)
    assert answer.answer == "Not found in the provided document."
    assert answer.citations == []


async def test_concurrency_deduplication_order_and_usage(settings):
    provider = FakeProvider(delay=0.025)
    settings.llm_concurrency = 2
    service = QAService(provider, settings)
    questions = ["Provider?", "Region?", "Provider?", "Annual revenue?", "Location?"]
    answers, usage = await service.run(CHUNKS, questions)
    assert [a.question for a in answers] == questions
    assert answers[3].status == "not_found"
    assert provider.peak == 2
    assert len(provider.calls) == 4
    assert provider.embed_calls == 1
    assert usage.llm_calls == 4 and usage.embedding_tokens == 123


async def test_global_concurrency_across_requests(settings):
    provider = FakeProvider(delay=0.02)
    settings.llm_concurrency = 2
    service = QAService(provider, settings)
    await asyncio.gather(
        service.run(CHUNKS, ["One", "Two", "Three"]), service.run(CHUNKS, ["Four", "Five", "Six"])
    )
    assert provider.peak == 2


async def test_timeout_is_per_question(settings):
    provider = FakeProvider(delay=0.05)
    settings.answer_timeout = 0.005
    answers, _ = await QAService(provider, settings).run(CHUNKS, ["Provider?"])
    assert answers[0].status == "error"
    assert answers[0].error_code == "provider_timeout"
    assert provider.active == 0


async def test_partial_provider_failure(settings):
    class MixedProvider(FakeProvider):
        async def answer(self, question, chunks):
            if question == "Fail":
                raise ProviderFault("provider_limit", "Try later")
            return await super().answer(question, chunks)

    answers, _ = await QAService(MixedProvider(), settings).run(CHUNKS, ["Fail", "Provider?"])
    assert [a.status for a in answers] == ["error", "answered"]


async def test_token_budget_before_provider_call(settings):
    provider = FakeProvider()
    settings.max_embedding_tokens = 100
    with pytest.raises(ClientFault, match="embedding_limit"):
        await QAService(provider, settings).run([Chunk("c1", "word " * 200, "page 1")], ["Q?"])
    assert provider.embed_calls == 0


async def test_unexpected_failure_cancels_siblings(settings):
    class BrokenProvider(FakeProvider):
        async def answer(self, question, chunks):
            if question == "Break":
                await asyncio.sleep(0.01)
                raise RuntimeError("failure")
            return await super().answer(question, chunks)

    provider = BrokenProvider(delay=0.2)
    with pytest.raises(ExceptionGroup):
        await QAService(provider, settings).run(CHUNKS, ["Break", "Slow"])
    assert provider.active == 0


def test_hybrid_retrieval_and_isolation():
    one = HybridIndex(
        [Chunk("a", "AWS hosting", "page 1"), Chunk("b", "Backups daily", "page 2")],
        [[1, 0], [0, 1]],
    )
    two = HybridIndex([Chunk("x", "Azure hosting", "page 1")], [[1, 0]])
    assert one.search("hosting", [1, 0], 1)[0].id == "a"
    assert two.search("hosting", [1, 0], 1)[0].text == "Azure hosting"


def test_presentation_quotes_are_normalized_but_paraphrases_are_rejected():
    result = ModelAnswer(
        status="answered",
        answer="AWS",
        evidence=[Evidence(chunk_id="c0001", quote='"Production is hosted on AWS in Oregon."')],
    )
    answer = validate_answer("Provider?", result, CHUNKS)
    assert answer.citations[0].quote == CHUNKS[0].text
    result.evidence[0].quote = '"Production is hosted on Azure in Oregon."'
    with pytest.raises(ProviderFault):
        validate_answer("Provider?", result, CHUNKS)


def test_short_verbatim_facts_are_valid_evidence():
    result = ModelAnswer(
        status="answered", answer="AWS", evidence=[Evidence(chunk_id="c0001", quote="AWS")]
    )
    assert validate_answer("Provider?", result, CHUNKS).citations[0].quote == "AWS"
