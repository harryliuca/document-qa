# Design and evaluation target

Source: [Engineering Onsite Challenge — Evaluation](https://docs.google.com/document/d/12xtmYEfdtzc2g_VdZfhE4Ia_iI4gHoQoNQz3JHCQVII/edit?tab=t.tf14f6ruqfrz), read October 8, 2026. This implementation targets the documented rubric; it does not claim an evaluator's score.

| Category | Points | Implementation | Verification evidence |
|---|---:|---|---|
| Backend correctness | 15 | Multipart API, PDF/JSON source support, multiple questions, ordered valid JSON | `tests/test_api.py`, real-format fixtures, live JSON/PDF checks |
| Error handling and robustness | 15 | Admission, byte/page/text/token/depth/chunk limits, killable parser, upload/answer/request deadlines, sanitized errors | `test_ingestion.py`, `test_boundary.py`, endpoint failure cases |
| Code quality and structure | 15 | Transport, configuration, ingestion, retrieval, provider adapter, service, and schemas are separate | `app/`, lint/format checks, this design rationale |
| Tests | 15 | Unit tests and backend integration tests use mocked AI; SDK contract tested with mock HTTP | `tests/`, pytest coverage, CI |
| Performance and concurrency | 15 | Async SDK calls, bounded shared semaphore, one embedding batch, duplicate reuse, CPU work off event loop | Cross-request concurrency and deduplication tests |
| Containerization and observability | 10 | Non-root Docker image, Compose limits, JSON timing/token/count logs, request IDs, health endpoint | Config inspection, log tests; Docker execution unavailable locally |
| Grounding and answer quality | 10 | Semantic + keyword retrieval, schema validation, exact evidence quotes, source locations, abstention | Quote-validation tests, supported/unsupported live fixture questions |
| Minimal frontend | 5 | Two file inputs, progress, result status, expandable citations, JSON download | Browser verification of actual uploads and results |

The rubric's performance sentence ends after “and”; no unprovided criterion was assumed. Demo video is optional. The requested submission artifact is a GitHub repository link; publishing and emailing are outside this local implementation.

## Decisions

**Keep the model fixed.** The brief explicitly requests `gpt-4o-mini`. Embedding uses `text-embedding-3-small`; it is a separate embedding operation, not a substitute answer-generation model.

**Use a small framework surface.** LangChain owns source-aware document splitting. The service coordinates typed Python components and the official async OpenAI SDK. A larger agent graph or tool loop adds latency and failure modes without helping this fixed upload/retrieve/answer task.

**Keep retrieval request-scoped.** Documents are uploaded per batch. A local vector index plus BM25 is sufficient for the capped document size, avoids operating an external database, and prevents cross-request source mixing. The tradeoff is re-embedding when the same source is uploaded in a later request. There is no unauthenticated shared cache.

**Separate missing evidence from execution errors.** Unsupported questions return `not_found`; API failures and invalid citations return `error`. Clients can distinguish knowledge gaps from a failed attempt. One question's expected failure does not discard successful answers for the rest of the batch.

**Preserve answer order and duplicates.** A result array can represent repeated questions; a question-keyed JSON object cannot. Duplicate work is coalesced internally, preserving all user-visible entries.

**Validate citations without claiming they prove truth.** Source-quote membership rejects nonexistent evidence. It does not prove that the quote entails the answer, that all relevant context was retrieved, or that an injected document cannot sway the model. The UI exposes evidence for inspection. The optional full-context strategy adds a separate model judge and targeted retry; it remains fallible and is not treated as a correctness oracle.

**Spend deliberately.** A retrieval request has one batched embedding call and at most one generation call per distinct question. Full-context requests have one generation per batch, one review per answered batch, and at most one individual generation/review retry per unresolved question. Limits bound input and output, no unbounded retry loops run, and usage is visible. There is no claim that local token limits enforce the shared $5 account budget.

## Module map

- `config.py`: validated settings and secret wrappers.
- `middleware.py`: admission, HTTP-body bounds, deadlines, correlation IDs, safe logging.
- `ingestion.py`: strict JSON, PDF extraction, document locations, LangChain chunking.
- `retrieval.py`: isolated vector and BM25 indices, rank fusion.
- `provider.py`: async OpenAI calls, structured response schema, safe provider errors.
- `service.py`: token budget, batched embeddings, concurrency, deduplication, evidence validation.
- `main.py`: lifecycle, multipart endpoint, error envelopes, health, static UI.
- `static/`: dependency-free browser UI; untrusted content uses `textContent`, never HTML insertion.

## Source references

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): JSON-schema constrained model responses and refusal handling.
- [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini): requested answer model.
- [LangChain text splitters](https://reference.langchain.com/python/langchain-text-splitters): source-aware chunking.
- [AnyIO worker processes](https://anyio.readthedocs.io/en/stable/subprocesses.html): cancellation terminates parser workers.

## Full-context alternative

`full_context_provider.py` constructs the stable source prefix and Structured Outputs schemas. `batching.py` plans topic-adjacent groups, checks result IDs and quote provenance, reviews semantic support, and runs one individual retry. Both strategies share the same provider semaphore. The model remains GPT-4o-mini.

Cache reuse reduces input processing cost only when the provider reports a hit. It does not remove document tokens from context-window accounting, establish a warm cache before concurrent requests, or eliminate answer-generation and verification latency. The trusted instructions, document serialization, and schema are identical across groups and retries; changing questions and retry hints come last.

The extra review call is deliberate: a real source quote can still support the opposite of a generated answer. Review uses entire cited chunks and location context, but does not rescan uncited document sections. False abstentions are possible because valid `not_found` answers are accepted without a second search. Topic grouping is deterministic keyword sorting, not a learned classifier; short groups are coalesced to avoid wasted singleton calls.

The 60K-token synthetic comparison passed all ten expected-answer checks in both modes. It did not establish superior accuracy, latency, or cost for full context, so retrieval remains the default. Representative labeled SOC 2 reports, repeated isolated cold/warm trials, and load tests are still needed before changing that default.
