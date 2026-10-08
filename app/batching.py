"""Bounded full-context batching with deterministic checks and one evidence-focused retry."""

import asyncio
from collections import Counter

import anyio

from app.config import Settings
from app.full_context_provider import (
    BatchProvider,
    batch_messages,
    request_tokens,
    review_messages,
    serialize_sources,
)
from app.models import (
    Answer,
    BatchOutput,
    Chunk,
    ClientFault,
    ModelAnswer,
    ProviderFault,
    ReviewOutput,
    Usage,
)
from app.retrieval import tokenize
from app.service import validate_answer

TOPICS = [
    {"iam", "access", "authentication", "password", "mfa", "identity", "permissions"},
    {"encryption", "encrypt", "encrypted", "tls", "keys", "cryptography"},
    {"bcp", "backup", "backups", "recovery", "continuity", "rto", "rpo", "disaster"},
    {"incident", "incidents", "notification", "breach", "response"},
    {"privacy", "personal", "pii", "retention", "gdpr"},
]
RETRYABLE = {"answer_too_long", "invalid_model_output"}


def topic(question: str) -> int:
    words = set(tokenize(question))
    scores = [len(words & terms) for terms in TOPICS]
    return scores.index(max(scores)) if max(scores) else len(TOPICS)


def plan_batches(questions: list[str], batch_size: int) -> tuple[list[dict], list[list[dict]]]:
    unique = [
        {"question_id": f"q{i + 1:04d}", "question": q}
        for i, q in enumerate(dict.fromkeys(questions))
    ]
    ordered = sorted(unique, key=lambda q: topic(q["question"]))
    # Keep related questions adjacent but coalesce short groups to avoid singleton batches.
    return unique, [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def index_results(items, expected: list[str]) -> tuple[dict, dict]:
    counts = Counter(item.question_id for item in items)
    unknown = set(counts) - set(expected)
    by_id, problems = {}, {}
    for qid in expected:
        if unknown:
            problems[qid] = "unexpected_question_ids"
        elif counts[qid] != 1:
            problems[qid] = "missing_or_duplicate_question_id"
        else:
            by_id[qid] = next(item for item in items if item.question_id == qid)
    return by_id, problems


def focused_ids(question: str, chunks: list[Chunk], limit=6) -> list[str]:
    terms = set(tokenize(question))
    ranked = sorted(
        chunks, key=lambda c: len(terms & set(tokenize(c.location + " " + c.text))), reverse=True
    )
    return [c.id for c in ranked[:limit]]


class FullContextService:
    def __init__(self, provider: BatchProvider, settings: Settings, slots: asyncio.Semaphore):
        self.provider, self.settings, self.slots = provider, settings, slots

    async def run(self, chunks: list[Chunk], questions: list[str], cache_mode="parallel"):
        unique, batches = plan_batches(questions, self.settings.full_context_batch_size)
        source_payload = serialize_sources(chunks)
        usage = Usage()

        async def budget(messages, schema):
            count = await anyio.to_thread.run_sync(request_tokens, messages, schema)
            if count > self.settings.full_context_input_tokens:
                raise ClientFault(
                    "full_context_limit",
                    "Document exceeds the full-context budget. "
                    "Choose retrieval mode or split the document.",
                    413,
                )

        # Reject before any paid calls. Never silently truncate the document or switch strategy.
        for batch in batches:
            await budget(batch_messages(source_payload, batch, {}), BatchOutput)

        def add_usage(delta):
            for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
                setattr(usage, key, getattr(usage, key) + getattr(delta, key))
            usage.usage_complete = usage.usage_complete and delta.usage_complete

        async def call(kind, operation):
            try:
                async with self.slots:
                    usage.llm_calls += 1
                    usage.verification_calls += int(kind == "review")
                    async with asyncio.timeout(self.settings.full_context_call_timeout):
                        result, delta = await operation()
                    add_usage(delta)
                    return result
            except TimeoutError:
                usage.usage_complete = False
                raise ProviderFault("provider_timeout", "The AI service timed out.") from None
            except ProviderFault as exc:
                if hasattr(exc, "usage"):
                    add_usage(exc.usage)
                else:
                    usage.usage_complete = False
                raise

        async def attempt(batch, focus, number):
            expected = [q["question_id"] for q in batch]
            question_map = {q["question_id"]: q["question"] for q in batch}
            await budget(batch_messages(source_payload, batch, focus), BatchOutput)
            try:
                result = await call(
                    "generate",
                    lambda: self.provider.generate_batch(
                        source_payload, batch, focus, min(12000, 300 + 750 * len(batch))
                    ),
                )
            except ProviderFault as exc:
                if exc.code in RETRYABLE:
                    return {}, dict.fromkeys(expected, exc.code)
                return {
                    qid: Answer(
                        question=question_map[qid],
                        status="error",
                        answer=exc.message,
                        error_code=exc.code,
                        attempts=number,
                    )
                    for qid in expected
                }, {}
            returned, pending = index_results(result.results, expected)
            answers, candidates = {}, []
            for qid, item in returned.items():
                if item.status == "needs_review":
                    pending[qid] = "ambiguous"
                    continue
                if item.status == "not_found" and item.evidence:
                    pending[qid] = "inconsistent_not_found"
                    continue
                try:
                    answer = validate_answer(
                        question_map[qid],
                        ModelAnswer(status=item.status, answer=item.answer, evidence=item.evidence),
                        chunks,
                    )
                except ProviderFault:
                    pending[qid] = "invalid_source_evidence"
                    continue
                answer.attempts = number
                answers[qid] = answer
                if answer.status == "answered":
                    cited = {c.chunk_id for c in answer.citations}
                    candidates.append(
                        {
                            "question_id": qid,
                            "question": question_map[qid],
                            "answer": answer.answer,
                            "citations": [c.model_dump() for c in answer.citations],
                            "sources": [
                                {"chunk_id": c.id, "location": c.location, "text": c.text}
                                for c in chunks
                                if c.id in cited
                            ],
                        }
                    )
            if candidates:
                # The reviewer sees full cited chunks, not just potentially cherry-picked quotes.
                try:
                    await budget(review_messages(candidates), ReviewOutput)
                    reviewed = await call("review", lambda: self.provider.review_batch(candidates))
                    verdicts, review_errors = index_results(
                        reviewed.results, [c["question_id"] for c in candidates]
                    )
                    for qid, verdict in verdicts.items():
                        if verdict.verdict == "supported":
                            answers[qid].evidence_check = "model_checked"
                        else:
                            pending[qid] = verdict.verdict
                    pending.update(review_errors)
                except (ProviderFault, ClientFault) as exc:
                    # Do not repeatedly spend on an unavailable reviewer; drafts stay withheld.
                    for candidate in candidates:
                        qid = candidate["question_id"]
                        answers[qid] = Answer(
                            question=question_map[qid],
                            status="needs_review",
                            answer="Evidence review could not complete. No verified answer is available.",  # noqa: E501
                            error_code=exc.code,
                            attempts=number,
                        )
            for qid in pending:
                answers.pop(qid, None)
            return answers, pending

        async def process(batch):
            answers, pending = await attempt(batch, {}, 1)
            if pending:

                async def retry(q):
                    qid = q["question_id"]
                    usage.retried_questions += 1
                    focus = {
                        "reason": pending[qid],
                        "focus_chunk_ids": focused_ids(q["question"], chunks),
                    }
                    try:
                        retried, unresolved = await attempt([q], focus, 2)
                    except ClientFault:
                        retried, unresolved = {}, {qid: "retry_context_limit"}
                    if qid in unresolved:
                        return qid, Answer(
                            question=q["question"],
                            status="needs_review",
                            answer="Unable to verify a supported answer after one focused retry.",
                            error_code=unresolved[qid],
                            attempts=2,
                        )
                    return qid, retried[qid]

                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(retry(q)) for q in batch if q["question_id"] in pending
                    ]
                answers.update(task.result() for task in tasks)
            return answers

        answered = {}
        if cache_mode == "warm_first" and len(batches) > 1:
            # Useful work, not an extra cache-warming charge. No cache-hit guarantee.
            answered.update(await process(batches.pop(0)))
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(process(batch)) for batch in batches]
        for task in tasks:
            answered.update(task.result())
        by_question = {q["question"]: answered[q["question_id"]] for q in unique}
        return [by_question[q].model_copy(deep=True) for q in questions], usage
