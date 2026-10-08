# Document QA

Upload a JSON list of questions and a PDF or JSON document. Receive one structured result per question with supporting source quotes, or an explicit `not_found`. Built for the Zania Engineering Onsite Challenge and its 100-point evaluation rubric.

## Run locally

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is selected automatically.

```sh
uv sync --frozen
cp .env.example .env.local  # Only if .env.local does not already exist.
# Set OPENAI_API_KEY in .env.local using your editor.
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

Open http://127.0.0.1:8000. Upload `fixtures/policy.json` (or `fixtures/policy.pdf`) and `fixtures/questions.json`. The first three questions have supported answers; the annual revenue question should return `not_found`. These fixtures contain fictional data.

The authorized challenge key is configured in this local checkout only. `.env.local` is ignored by Git and Docker and must not accompany a submission. The application never sends the key to the browser. A recipient must supply their own authorized key.

The first local launch downloads the public `cl100k_base` tokenizer vocabulary; subsequent runs use its cache. `TIKTOKEN_CACHE_DIR` can point at a prepopulated cache for offline environments. Docker preloads it at build time. Tests use mocked AI calls and require no API key or paid API requests.

## Docker

```sh
docker compose up --build
```

Compose binds to localhost, runs as a non-root user with a read-only filesystem, and sets memory, CPU, process, and temporary-storage limits. Docker was unavailable on the implementation machine; the image is supplied but its build/run is not locally verified. CI includes an image build.

## API

```sh
curl --fail-with-body http://127.0.0.1:8000/api/answers \
  -F 'document=@fixtures/policy.json;type=application/json' \
  -F 'questions=@fixtures/questions.json;type=application/json'
```

- `POST /api/answers`: multipart file fields `document` and `questions`, exactly once each.
- `GET /health`: process liveness and whether an AI provider is configured; this does not verify key validity or quota.
- `GET /openapi.json`: machine-readable API schema.
- `GET /`: self-contained upload UI, with evidence expansion, JSON download, and a copyable raw JSON view.

Questions accept either `["Question one?", "Question two?"]` or `{"questions":["Question one?"]}`. They must be nonempty strings. Source JSON accepts nested objects, arrays, and scalar values; object keys are preserved in text and locations use escaped JSON pointers. Flat records within arrays remain intact, preserving question/answer relationships. Duplicate JSON object keys and non-finite numbers are rejected.

```json
{
  "request_id": "example-id",
  "results": [
    {
      "question": "Which cloud provider hosts the service?",
      "status": "answered",
      "answer": "Amazon Web Services (AWS).",
      "citations": [
        {
          "chunk_id": "c0002",
          "location": "JSON pointer /hosting",
          "quote": "The production service runs on Amazon Web Services (AWS) in us-west-2 (Oregon)."
        }
      ],
      "error_code": null
    }
  ],
  "usage": {"embedding_tokens": 164, "input_tokens": 520, "output_tokens": 80, "llm_calls": 1},
  "duration_ms": 1800,
  "document_chunks": 5
}
```

The example is illustrative; captured real responses are in `validation/`. The challenge's sample output notation is not valid JSON, so the contract uses an ordered results array. This preserves repeated questions and their original order rather than losing duplicates in an object map.

`status` is `answered`, `not_found`, `needs_review`, or `error`. Full-context mode withholds unverified drafts as `needs_review`. A successful HTTP 200 can contain per-question errors; clients must inspect each result. Invalid citations are reported as `error/ungrounded_answer`, never relabeled as missing source information. Batch-wide parsing/indexing failures return non-2xx JSON:

```json
{"error":{"code":"invalid_pdf","message":"The uploaded file is not a valid PDF."},"request_id":"example-id"}
```

Status codes: 400 malformed multipart; 401 missing/invalid optional access token; 403 cross-site browser request; 408 slow upload; 413 resource limit; 415 unsupported file type; 422 invalid content; 429 server busy; 502/503 AI failure/configuration; 504 overall request deadline. Unexpected errors use a sanitized 500 response. No raw provider error, upload text, filename, or key is returned in errors/logs.

## Full-document batching (development)

Select **Full document · topic batches** in the UI, or call:

```sh
curl --fail-with-body 'http://127.0.0.1:8000/api/answers?strategy=full_context_batch&cache_mode=parallel' \
  -F 'document=@fixtures/policy.pdf' \
  -F 'questions=@fixtures/questions.json'
```

Retrieval remains the default. Full-context mode accepts up to 75 questions, groups related questions into batches of five, and shares the four-call concurrency limit with retrieval requests. It makes no embedding calls. Stable IDs preserve original question order and duplicates.

Every generation sees the complete extracted document as an **untrusted user message**, following trusted system instructions. Its stable prefix and document-derived `prompt_cache_key` allow automatic provider caching; cache hits are not guaranteed. `cache_mode=warm_first` completes the first useful batch before launching the rest. It adds no dummy warming request but may increase latency.

Returned IDs and source quotes are checked. A separate model call reviews claim support using the full cited chunks. Missing/duplicate IDs, invalid evidence, conflicting or incomplete answers get one individual retry with focused source hints and the same full document. Unresolved drafts are withheld as `needs_review`. Valid `not_found` answers and provider outages do not trigger retries. A model review can still miss errors, conflicts elsewhere in the document, or false abstentions; `model_checked` is not a correctness guarantee.

The response includes `strategy`, `cache_mode`, per-answer `attempts` and `evidence_check`, plus `cached_input_tokens`, `verification_calls`, `retried_questions`, and `usage_complete`. `llm_calls` counts both generation and review. Cached tokens are a subset of input tokens. Interrupted calls can be billed without complete usage reporting.

| Full-context setting | Default |
|---|---:|
| `MAX_FULL_CONTEXT_QUESTIONS` | 75 |
| `FULL_CONTEXT_BATCH_SIZE` | 5 (maximum 10) |
| `FULL_CONTEXT_INPUT_TOKENS` | 100,000, including schema and framing estimate |
| `FULL_CONTEXT_CALL_TIMEOUT` | 60 seconds |
| `FULL_CONTEXT_REQUEST_TIMEOUT` | 300 seconds |

An oversized document returns `413/full_context_limit` before generation. It is never silently truncated or switched to retrieval. Existing upload, extraction, page, and chunk limits also apply. “Full document” refers to all extracted chunks, not a guarantee of lossless PDF extraction.

The opt-in comparison runner is `python -m scripts.compare_strategies --live`. It uses the configured key, makes paid calls, and checks a conservative run estimate against `--max-usd` (default $1). This is a per-run estimate, not an account spending limit. See [the comparison report](validation/FULL_CONTEXT.md) for observed latency, cost, cache hits, and limitations.

## Design

```mermaid
flowchart LR
  U[Two uploads] --> V[Admission and byte limits]
  V --> P[Cancellable parser process]
  P --> C[LangChain source-aware chunks]
  C --> E[Batch document and question embeddings]
  E --> R[Per-request hybrid index]
  R --> A[Bounded concurrent GPT-4o-mini answers]
  A --> G[Schema and source-quote validation]
  G --> J[Ordered JSON results with citations]
```

- **FastAPI** handles transport and response models. CPU-heavy PDF parsing runs in killable AnyIO worker processes; tokenization, indexing, and retrieval run outside the event loop.
- **LangChain** `Document` and `RecursiveCharacterTextSplitter` preserve source metadata while producing 1,400-character chunks with 180-character overlap. Pages never merge across PDF boundaries.
- **Retrieval:** a request-local NumPy cosine vector index and BM25 keyword index are combined by reciprocal-rank fusion. The top six chunks supply each answer. `text-embedding-3-small` produces 512-dimensional vectors. One embedding batch includes all chunks and distinct questions.
- **Generation:** `gpt-4o-mini` is fixed to the challenge requirement, using OpenAI Structured Outputs, temperature zero, and a 650-token output cap. Instructions require abstention and explicit disclosure of unsupported parts of multi-part questions. Uploaded content cannot call tools or execute code.
- **Evidence checks:** every cited ID must identify a retrieved chunk; each quote must occur in that chunk after whitespace normalization and optional removal of presentation quotation marks. Unknown IDs, invented quotes, and missing evidence are rejected. These checks establish quote provenance, not semantic entailment of every claim.
- **Concurrency:** the service shares four provider slots across all requests. Admission permits two active upload/answer requests per process. Duplicate questions share generation work, then expand back to the original order. `TaskGroup` cancels sibling work on unexpected failures; known per-question failures preserve other results.
- **Isolation:** documents, vectors, and answers exist only for the current request. There is no application-managed cross-user cache or persistent vector database. OpenAI may cache exact prompt prefixes within its own service. At this bounded scale, an in-memory index avoids database operations and additional infrastructure. Persistent document reuse would require authenticated ownership and an explicitly scoped store.

See [DESIGN.md](DESIGN.md) for the evaluation mapping and tradeoffs.

## Limits and cost control

| Limit | Default |
|---|---:|
| Total HTTP body | 11 MiB |
| Source document | 10 MiB |
| Questions file | 64 KiB |
| Questions / question length | 20 / 1,000 characters |
| PDF pages / extracted characters | 100 / 250,000 |
| JSON depth | 20 |
| Chunks / aggregate embedding input tokens | 300 / 80,000 |
| Upload / parser deadline | 20s / 15s |
| Per-answer / whole processing deadline | 30s / 120s |
| Simultaneous requests / provider calls | 2 / 4 per process |

Limits are configured in `app/config.py` and may be overridden by corresponding uppercase environment variables. The UI describes the shipped defaults. No automatic SDK retries are enabled. Full-context mode permits one targeted application retry per unresolved question. Token usage is returned and emitted in JSON logs, but it is not a financial ledger: failed or interrupted calls can still incur provider charges. The challenge's $5 quota is controlled by the provider; this app does not know the shared key's remaining balance or enforce an account-wide spending cap.

Retrieval mode sends extracted text for embeddings and retrieved excerpts for answers. Full-context mode sends the entire extracted document for each generation call and cited chunks for evidence review. Local multipart spooling may use temporary files, which are closed after parsing; the app does not persist uploads. If hosting beyond localhost, configure `APP_API_TOKEN` and an authenticated TLS gateway with per-user rate limits. Multiple workers multiply the in-process concurrency limits.

## Tests and checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=app --cov-report=term-missing
```

Tests cover real JSON/PDF parsing, password-required/scan rejection, AES PDFs that open without a password, limits, provenance, invalid quotes, request isolation, deduplication, bounded concurrency across requests, partial failures, upload/request deadlines, overload recovery, sanitized logs/errors, and real OpenAI SDK behavior using a mock HTTP transport. No live model is used by the test suite. A separate live fixture check and browser verification are documented in `VALIDATION.md`.

## Boundaries and next steps

- AES-encrypted PDFs that open without a password are supported via `pypdf[crypto]`. PDFs requiring an opening password are rejected with a specific message.
- Text extraction only: no OCR; scanned PDFs are rejected. Complex tables and mixed scanned/text pages may lose information.
- Retrieval can miss evidence, and a real quote can be misinterpreted by an LLM. “Not found” means unsupported by retrieved context, not proof of absence across every page. A labeled evaluation set is needed for quantitative answer-quality claims.
- Prompt-injection resistance is layered through data separation, no tools, and quote validation, but it is not a guarantee that the model cannot be influenced.
- The initial implementation is intentionally synchronous at the HTTP request level. A durable job queue and authenticated persisted documents are appropriate extensions if larger workloads are required.
- This implementation was designed, coded, and tested with Codex assistance. Human review should verify engineering choices and live answers before submission.
