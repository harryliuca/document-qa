import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.models import Chunk, ProviderFault
from app.provider import OpenAIProvider


def provider_with_transport(handler):
    provider = OpenAIProvider("test-key", "gpt-4o-mini", 5)
    provider.client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


async def test_real_sdk_structured_output_and_usage():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["max_completion_tokens"] == 650
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "status": "answered",
                                    "answer": "AWS",
                                    "evidence": [{"chunk_id": "c1", "quote": "Hosted on AWS"}],
                                }
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
            },
        )

    provider = provider_with_transport(handler)
    try:
        answer, usage = await provider.answer("Provider?", [Chunk("c1", "Hosted on AWS", "page 1")])
        assert answer.status == "answered"
        assert (usage.input_tokens, usage.output_tokens) == (40, 20)
    finally:
        await provider.close()


async def test_real_sdk_embeddings_reordered():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
            },
        )

    provider = provider_with_transport(handler)
    try:
        vectors, tokens = await provider.embed(["one", "two"])
        assert vectors == [[1.0, 0.0], [0.0, 1.0]] and tokens == 12
    finally:
        await provider.close()


@pytest.mark.parametrize(
    "status,code", [(401, "provider_auth"), (429, "provider_limit"), (500, "provider_unavailable")]
)
async def test_sdk_errors_sanitized(status, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, json={"error": {"message": "SENSITIVE PROVIDER DETAIL", "type": "test_error"}}
        )

    provider = provider_with_transport(handler)
    try:
        with pytest.raises(ProviderFault) as exc:
            await provider.embed(["private input"])
        assert exc.value.code == code
        assert "SENSITIVE" not in exc.value.message
        assert len(calls) == 1  # No automatic spending through SDK retries.
    finally:
        await provider.close()


async def test_model_refusal():
    provider = provider_with_transport(
        lambda request: httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": None, "refusal": "No"},
                    }
                ],
            },
        )
    )
    try:
        with pytest.raises(ProviderFault) as exc:
            await provider.answer("Q?", [])
        assert exc.value.code == "model_refusal"
    finally:
        await provider.close()


async def test_content_filter_becomes_per_question_failure():
    provider = provider_with_transport(
        lambda request: httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "content_filter",
                        "message": {"role": "assistant", "content": None},
                    }
                ],
            },
        )
    )
    try:
        with pytest.raises(ProviderFault) as exc:
            await provider.answer("Q?", [])
        assert exc.value.code == "model_refusal"
    finally:
        await provider.close()


async def test_batch_sdk_cache_prefix_location_and_usage():
    from app.full_context_provider import FullContextProvider, serialize_sources

    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        question = json.loads(body["messages"][-1]["content"])["questions"][0]
        return httpx.Response(
            200,
            json={
                "id": "batch",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "results": [
                                        {
                                            "question_id": question["question_id"],
                                            "status": "not_found",
                                            "answer": "",
                                            "evidence": [],
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 2048,
                    "completion_tokens": 30,
                    "total_tokens": 2078,
                    "prompt_tokens_details": {"cached_tokens": 1024},
                },
            },
        )

    base = provider_with_transport(handler)
    try:
        provider = FullContextProvider(base)
        source = serialize_sources([Chunk("c1", "Hosted on AWS", "$.production.cloud")])
        for qid in ["q1", "q2"]:
            result, usage = await provider.generate_batch(
                source, [{"question_id": qid, "question": "Where?"}], {}, 1200
            )
            assert result.results[0].question_id == qid
            assert usage.cached_input_tokens == 1024
        assert bodies[0]["messages"][:2] == bodies[1]["messages"][:2]
        assert bodies[0]["prompt_cache_key"] == bodies[1]["prompt_cache_key"]
        assert bodies[0]["store"] is False and bodies[0]["max_completion_tokens"] == 1200
        assert "$.production.cloud" in bodies[0]["messages"][1]["content"]
        assert bodies[0]["response_format"] == bodies[1]["response_format"]
    finally:
        await base.close()


@pytest.mark.parametrize(
    "variant,code",
    [
        ("length", "answer_too_long"),
        ("content_filter", "model_refusal"),
        ("refusal", "model_refusal"),
        ("invalid", "invalid_model_output"),
        ("outage", "provider_limit"),
    ],
)
async def test_batch_failures_sanitized_and_no_sdk_retry(variant, code):
    from app.full_context_provider import FullContextProvider

    calls = []

    def handler(request):
        calls.append(request)
        if variant == "outage":
            return httpx.Response(429, json={"error": {"message": "private", "type": "test"}})
        return httpx.Response(
            200,
            json={
                "id": "batch",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": variant
                        if variant in {"length", "content_filter"}
                        else "stop",
                        "message": {
                            "role": "assistant",
                            "content": "invalid" if variant == "invalid" else None,
                            "refusal": "declined" if variant == "refusal" else None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 20, "total_tokens": 2020},
            },
        )

    base = provider_with_transport(handler)
    try:
        with pytest.raises(ProviderFault) as exc:
            await FullContextProvider(base).generate_batch("{}", [], {}, 1000)
        assert exc.value.code == code
        assert len(calls) == 1
        if variant in {"length", "refusal"}:
            assert exc.value.usage.input_tokens == 2000
    finally:
        await base.close()
