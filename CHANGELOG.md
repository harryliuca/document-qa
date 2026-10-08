# Changelog

## 0.6.0.dev1 — unreleased

- Fixed parsing of AES-encrypted PDFs that open without a password; added the locked crypto dependency, preserved password-required rejection, and corrected misleading parser errors.

- Optional full-document topic batching with stable cache prefix, five-question groups, and parallel/warm-first scheduling.
- Exact question-ID checks, quote validation, separate claim-support review, one focused retry, and withheld `needs_review` results.
- Shared provider concurrency, input preflight, per-call/request deadlines, and cache/review/retry usage metrics.
- UI strategy selection, extended question limits, and evidence-check labels.
- Fixed JSON ancestor-path context in retrieval embeddings/model input and rejected overflowing JSON numbers.
- Added mocked batch/SDK failure tests and a paid opt-in 60K-token synthetic comparison runner.
- Retrieval remains the default; `v0.5` is preserved.

## 0.5.0 — 2026-10-08

Initial implementation snapshot, tagged `v0.5`.

- FastAPI document Q&A supporting PDF/JSON sources and multiple questions.
- LangChain chunking, hybrid retrieval, bounded concurrent GPT-4o-mini answers.
- Structured results with source quotes, abstentions, and per-question errors.
- Upload UI with evidence disclosure and raw JSON/export controls.
- Resource limits, timeouts, sanitized structured logs, and request isolation.
- Docker/Compose configuration, CI workflow, fixtures, and 61 automated tests.

### Known limitations from the implementation review

- JSON array records can lose ancestor-path context in model-visible evidence.
- Quote membership validation does not establish semantic support for an answer.
- Extreme JSON numeric exponents can overflow to non-finite Python values.
- Docker execution, representative load performance, and broad answer quality remain unverified locally.
- File saving through the in-app browser was not independently confirmed; raw JSON is available.

The review estimate was 88/100 against the challenge rubric, not an official evaluator score.
