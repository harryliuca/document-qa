"""Opt-in live smoke comparison, with synthetic gold facts and conservative spend ceiling.

Run from repository root: .venv/bin/python -m scripts.compare_strategies --live
The synthetic appendix is irrelevant high-token audit data, NOT a realistic SOC 2 evaluation.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from app.batching import FullContextService
from app.config import Settings
from app.full_context_provider import FullContextProvider, serialize_sources
from app.ingestion import parse_document
from app.provider import OpenAIProvider
from app.service import QAService, embedding_token_count

CASES = [
    ("Is MFA required for administrator access?", "answered", ["mfa", "required"]),
    ("How often are access reviews performed?", "answered", ["quarter"]),
    ("Are contractors permitted production access?", "answered", ["not"]),
    ("Which encryption protects stored backups?", "answered", ["aes-256"]),
    ("What is the minimum TLS version?", "answered", ["1.2"]),
    ("How long are backups retained?", "answered", ["30"]),
    ("What are the recovery time and recovery point objectives?", "answered", ["4", "1"]),
    (
        "Which cloud hosts production and which regions host primary and backup data?",
        "answered",
        ["aws", "oregon", "virginia"],
    ),
    ("What was annual revenue in 2025?", "not_found", []),
    ("Which cyber insurance provider is contracted?", "not_found", []),
]


def fixture():
    facts = {
        "identity": {
            "controls": [
                {
                    "name": "Administrator access",
                    "requirement": "MFA is required for all administrator access.",
                },
                {"name": "Access review", "frequency": "Access reviews are performed quarterly."},
                {"name": "Contractors", "rule": "Contractors are not permitted production access."},
            ]
        },
        "encryption": (
            "Stored backups use AES-256 encryption. TLS 1.2 is the minimum transport version."
        ),
        "continuity": (
            "Backups are retained for 30 days. Recovery time objective is 4 hours; "
            "recovery point objective is 1 hour."
        ),
        "hosting": (
            "Production runs on AWS. The primary data region is Oregon "
            "and the backup region is Virginia."
        ),
    }
    # Dispersed facts around unrelated records exercise document size and JSON hierarchy.
    appendix = [
        f"Synthetic audit record {i}: {hashlib.sha256(str(i).encode()).hexdigest()}"
        for i in range(1150)
    ]
    return {
        "identity": facts["identity"],
        "appendix_a": "\n".join(appendix[:400]),
        "encryption": facts["encryption"],
        "appendix_b": "\n".join(appendix[400:800]),
        "continuity": facts["continuity"],
        "appendix_c": "\n".join(appendix[800:]),
        "hosting": facts["hosting"],
    }


def estimated_cost(usage):
    # gpt-4o-mini and text-embedding-3-small published standard rates, USD per million.
    return (
        (usage.input_tokens - usage.cached_input_tokens) * 0.15
        + usage.cached_input_tokens * 0.075
        + usage.output_tokens * 0.60
        + usage.embedding_tokens * 0.02
    ) / 1_000_000


async def run(args):
    settings = Settings()
    raw = json.dumps(fixture()).encode()
    chunks = parse_document(raw, "json", {})
    source_tokens = embedding_token_count([serialize_sources(chunks)])
    # Bound every possible generation/review with the configured input caps and
    # output limits; includes retries and the baseline's ten generations.
    bound = (
        ((source_tokens + 6000) * 24 + settings.full_context_input_tokens * 24 + 128_000 * 10)
        * 0.15
        + 60_000 * 0.60
        + 80_000 * 0.02
    ) / 1_000_000
    if bound > args.max_usd:
        raise SystemExit(f"Conservative run estimate ${bound:.3f} exceeds --max-usd")
    print(
        f"Synthetic source: {len(chunks)} chunks, {source_tokens} tokens. Spend bound ${bound:.3f}",
        flush=True,
    )
    provider = OpenAIProvider(settings.openai_api_key.get_secret_value(), settings.model, 60)
    rag = QAService(provider, settings)
    full = FullContextService(FullContextProvider(provider), settings, rag.provider_slots)
    report = {
        "fixture": "Synthetic JSON controls and audit hashes; not a SOC 2 quality benchmark",
        "source_tokens": source_tokens,
        "source_chunks": len(chunks),
        "runs": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for strategy, mode in [
            ("retrieval", None),
            ("full_context_batch", "parallel"),
            ("full_context_batch", "warm_first"),
        ]:
            started = time.monotonic()
            questions = [c[0] for c in CASES]
            if mode:
                results, usage = await full.run(chunks, questions, mode)
            else:
                results, usage = await rag.run(chunks, questions)
            checks = [
                r.status == status and all(term in r.answer.lower() for term in terms)
                for r, (_, status, terms) in zip(results, CASES, strict=True)
            ]
            record = {
                "strategy": strategy,
                "cache_mode": mode,
                "duration_seconds": round(time.monotonic() - started, 2),
                "heuristic_checks_passed": sum(checks),
                "total": len(checks),
                "estimated_usd": round(estimated_cost(usage), 6),
                "usage": usage.model_dump(),
                "results": [r.model_dump() for r in results],
            }
            report["runs"].append(record)
            await asyncio.to_thread(output.write_text, json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "results"}), flush=True)
    finally:
        await provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize the paid API comparison")
    parser.add_argument("--max-usd", type=float, default=1.0)
    parser.add_argument("--output", default="validation/full-context-comparison.json")
    arguments = parser.parse_args()
    if not arguments.live:
        parser.error("Supply --live to make paid calls; offline tests use pytest.")
    asyncio.run(run(arguments))
