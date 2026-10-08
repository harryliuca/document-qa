# Full-context batching validation — October 8, 2026

Implementation: local `full-context-batch` branch, version `0.6.0.dev1`. Existing `v0.5` remains unchanged. No new release is published.

## Live comparison

The same deterministic synthetic JSON document contained dispersed policy facts, nested array records, and unrelated audit hashes: **60,009 serialized document tokens, 95 chunks, ten questions**. Eight supported questions covered IAM, negation, encryption, backups, recovery objectives, and multi-part hosting details; two asked for absent information. All calls used `gpt-4o-mini`.

| Run order | Strategy | Time | Estimated USD | AI calls (reviews) | Cached input / total input | Expected-answer checks |
|---|---|---:|---:|---:|---:|---:|
| 1 | Retrieval | 6.15 s | $0.004205 | 10 (0) | 0 / 18,636 | 10/10 |
| 2 | Full context, parallel | 6.88 s | $0.017881 | 4 (2) | 8,064 / 120,776 | 10/10 |
| 3 | Full context, warm-first | 7.70 s | $0.009566 | 4 (2) | 118,912 / 120,775 | 10/10 |

All runs reported complete usage and zero targeted retries. Retrieval also used 57,710 embedding tokens. Combined estimated cost was $0.031652, excluding the separate small UI smoke test. Costs use standard published model rates and are estimates, not billing reconciliation.

The third run reused the same document after the second run, so its high cache hit rate **does not isolate the effect of warm-first scheduling**. There was one trial per configuration, no guaranteed cold-cache reset, no statistically meaningful latency comparison, and no load test. The checks compare expected statuses and answer terms; manual inspection of returned answers confirmed the intended negation and multi-part facts. They are not an independent semantic accuracy score. The appendix deliberately tests context size, not realistic SOC 2 language complexity.

[Raw comparison results](full-context-comparison.json) contain responses, quotes, usage, and timings. Reproduce with `python -m scripts.compare_strategies --live`; this makes paid API calls. The runner's conservative estimate includes possible targeted retries; `--max-usd` rejects an excessive estimate before API calls. The app itself does not enforce the shared account's remaining balance.

## Browser PDF smoke test

Uploaded the existing fictional `fixtures/policy.pdf` and four-question JSON through the UI at localhost:8766, selected full-document parallel batching, and verified:

- Three supported answers with page-1 citations and `model_checked` labels.
- One legitimate `not_found` for annual revenue.
- 3.503 seconds, two AI calls including one review, no embeddings, no retries, 1,818 input tokens, 274 output tokens, zero cached tokens.
- Strategy controls, 75-question hint, live progress, results, evidence disclosure, and raw JSON present; layout visually inspected.

The challenge-linked sample PDF URL returned HTTP 200 but failed PDF-content validation, so that external report was not used. Existing PDF parsing tests and this local fixture passed; representative SOC 2 answer quality remains unmeasured.

## Recommendation and limits

Keep retrieval as the default. Full context is a selectable alternative that avoids retrieval omissions within the extracted input, with bounded individual retries and explicit unresolved results. The synthetic trial shows working caching and fewer calls, but does not show a quality improvement or lower cost than retrieval.

Model-based review remains fallible, reads cited chunks rather than every other section, and does not independently validate `not_found`. Real PDF extraction can lose tables or scanned content. Docker execution remains unverified locally.

Sources for API behavior and estimates: [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), and [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small).

## Automated verification

87 tests passed; application statement coverage was 96%. New tests exercise topic grouping, duplicate expansion and ordering, malformed question IDs, one-retry limits, invalid quotes, contradicted/incomplete answers, legitimate abstention, provider timeout/outage, review failure, input preflight, concurrency, full-mode endpoint limits, JSON numeric overflow, stable cache prefixes, and real SDK refusal/truncation/error handling via mock HTTP. Lint, Python formatting, JavaScript syntax, frontend formatting, and Git whitespace checks passed. See [test output](full-context-tests.txt).
