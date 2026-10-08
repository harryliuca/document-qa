# Changelog

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
