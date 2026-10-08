import asyncio
import re

import anyio
import tiktoken
from openai import LengthFinishReasonError

from app.config import Settings
from app.models import Answer, Chunk, Citation, ClientFault, ModelAnswer, ProviderFault, Usage
from app.provider import Provider
from app.retrieval import HybridIndex


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_answer(question: str, result: ModelAnswer, sources: list[Chunk]) -> Answer:
    if result.status == "not_found":
        return Answer(
            question=question, status="not_found", answer="Not found in the provided document."
        )
    by_id = {c.id: c for c in sources}
    citations = []
    if not result.answer.strip() or not 1 <= len(result.evidence) <= 4:
        raise ProviderFault(
            "ungrounded_answer", "Could not verify supporting evidence. Please retry."
        )
    for evidence in result.evidence:
        source = by_id.get(evidence.chunk_id)
        quote = normalized(evidence.quote)
        # Models sometimes wrap a verbatim excerpt in presentation quotation marks.
        # Remove only a matching outer pair, then still require an exact source substring.
        quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if source and quote not in normalized(source.text):
            if len(quote) >= 2 and quote_pairs.get(quote[0]) == quote[-1]:
                quote = quote[1:-1].strip()
        if source is None or not quote or quote not in normalized(source.text):
            raise ProviderFault(
                "ungrounded_answer", "Could not verify supporting evidence. Please retry."
            )
        citations.append(Citation(chunk_id=source.id, location=source.location, quote=quote))
    return Answer(
        question=question,
        status="answered",
        answer=result.answer,
        citations=citations,
        evidence_check="quote_checked",
    )


def embedding_token_count(texts: list[str]) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(t, disallowed_special=())) for t in texts)


class QAService:
    def __init__(self, provider: Provider, settings: Settings):
        self.provider, self.settings = provider, settings
        # Shared across all requests handled by this application instance.
        self.provider_slots = asyncio.Semaphore(settings.llm_concurrency)

    async def run(self, chunks: list[Chunk], questions: list[str]) -> tuple[list[Answer], Usage]:
        unique = list(dict.fromkeys(questions))
        texts = [f"{c.location}\n{c.text}" for c in chunks] + unique
        tokens = await anyio.to_thread.run_sync(embedding_token_count, texts)
        if tokens > self.settings.max_embedding_tokens:
            raise ClientFault(
                "embedding_limit", "Document exceeds the token budget. Split it into parts.", 413
            )
        usage = Usage()
        # One batched embedding request for the document and all distinct questions.
        async with self.provider_slots:
            vectors, usage.embedding_tokens = await self.provider.embed(texts)
        if len(vectors) != len(texts):
            raise ProviderFault(
                "invalid_embeddings", "The AI service returned incomplete embeddings."
            )
        index = await anyio.to_thread.run_sync(HybridIndex, chunks, vectors[: len(chunks)])

        async def answer_one(position: int, question: str) -> Answer:
            sources = await anyio.to_thread.run_sync(
                index.search, question, vectors[len(chunks) + position], self.settings.top_k
            )
            try:
                async with self.provider_slots:
                    async with asyncio.timeout(self.settings.answer_timeout):
                        usage.llm_calls += 1
                        result, delta = await self.provider.answer(question, sources)
                        usage.input_tokens += delta.input_tokens
                        usage.output_tokens += delta.output_tokens
                        usage.cached_input_tokens += delta.cached_input_tokens
                        usage.usage_complete = usage.usage_complete and delta.usage_complete
                return validate_answer(question, result, sources)
            except TimeoutError:
                usage.usage_complete = False
                fault = ProviderFault("provider_timeout", "This question timed out. Please retry.")
            except LengthFinishReasonError:
                usage.usage_complete = False
                fault = ProviderFault(
                    "answer_too_long", "Answer exceeded the output limit. Narrow the question."
                )
            except ProviderFault as exc:
                usage.usage_complete = False
                fault = exc
            return Answer(
                question=question, status="error", answer=fault.message, error_code=fault.code
            )

        # TaskGroup cancels sibling work on unexpected errors or request cancellation.
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(answer_one(i, q)) for i, q in enumerate(unique)]
        by_question = {q: task.result() for q, task in zip(unique, tasks, strict=True)}
        return [by_question[q].model_copy() for q in questions], usage
