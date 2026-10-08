# Validation record

Verified locally on October 8, 2026 with Python 3.12.12. This is implementation evidence, not an assigned rubric score.

## Automated checks

- **61 tests passed; 97% backend statement coverage** (`validation/test-results.txt`).
- Ruff lint and formatting checks passed.
- `node --check app/static/app.js` passed.
- Tests use mocked providers or mock HTTP transport; no API key is needed for paid services.
- Repository source scan found no API-key patterns outside the ignored local environment file. That file has mode `0600`; Git ignores it and the virtual environment. Docker excludes all environment files.

Coverage includes source parsing, invalid JSON/PDF, duplicate JSON keys, source locations, blank/scanned/encrypted PDF cases, page/byte/text/chunk/depth/token limits, request isolation, citation ID/quote validation, short valid evidence, normalization of presentation quotes, stable order, deduplication, bounded concurrency across requests, partial failures, cancellation, upload/request deadlines, admission recovery, authentication, sanitized logs/errors, SDK structured output, refusals, and API error translation.

## Live API checks

Using the user-authorized challenge key with `gpt-4o-mini`, and `text-embedding-3-small` for retrieval, each fixture request returned HTTP 200 and statuses:

`answered`, `answered`, `answered`, `not_found`

| Fixture | Processing time | Embedding tokens | Input tokens | Output tokens | Generation calls |
|---|---:|---:|---:|---:|---:|
| JSON | 1,931 ms | 164 | 2,110 | 218 | 4 |
| PDF | 1,534 ms | 161 | 1,894 | 210 | 4 |

Captured responses: `validation/live-json-result.json` and `validation/live-pdf-result.json`. Checks confirmed the 24-hour notification SLA, 30-day retention, AWS regions, and abstention for unspecified revenue. The fixtures are small fictional documents; these timings are not a load-test benchmark. The remaining balance of the shared key was not available and is not claimed.

The first live check exposed presentation quotation marks around an exact excerpt. Validation now removes a matching outer quote pair if necessary, while still requiring verbatim source membership. A regression test confirms paraphrases are rejected. Short literal facts such as “AWS” are also accepted when present in the source.

## Browser checks

- Uploaded both JSON and PDF fixtures through the actual two-file UI.
- Saw progress transition to three supported answers and one “not found” result.
- Expanded source evidence and verified location plus quote rendering.
- Inspected the desktop layout and browser console; no JavaScript errors were reported.
- The JSON export link is implemented as a native `download` anchor to a Blob. The in-app browser automation did not report a download event, so physical file saving is **not independently confirmed** here. A raw JSON disclosure is included as a copyable fallback; the API responses are also saved in `validation/`.

## Verification limits

Docker is not installed on this machine. The Dockerfile and Compose configuration are included, and CI includes an image build, but neither Docker execution nor remote CI has been observed. Mobile viewport behavior, large-document load performance, and statistical answer quality on a labeled real-world corpus have not been measured. The external sample SOC 2 PDF and spreadsheet linked in the challenge were not used in these fixture checks.

Source quotes are checked for existence, not logical entailment of every generated claim. Review cited evidence; no automated score or production-readiness certification is claimed.
