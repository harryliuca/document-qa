"""Untrusted document parsing; run in a cancellable worker process."""

import io
import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings
from app.models import Chunk, ClientFault


def strict_json(raw: bytes):
    def reject_constant(value):
        raise ValueError("Non-finite number")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate object key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8-sig"), parse_constant=reject_constant, object_pairs_hook=unique_pairs
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ClientFault(
            "invalid_json", "Upload valid UTF-8 JSON without duplicate keys."
        ) from None


def parse_questions(raw: bytes, settings: Settings) -> list[str]:
    if len(raw) > settings.max_questions_bytes:
        raise ClientFault("questions_too_large", "Questions file exceeds 64 KB.", 413)
    value = strict_json(raw)
    if isinstance(value, dict) and set(value) == {"questions"}:
        value = value["questions"]
    if not isinstance(value, list) or not 1 <= len(value) <= settings.max_questions:
        raise ClientFault(
            "invalid_questions", f"Supply a JSON array of 1–{settings.max_questions} questions."
        )
    if any(
        not isinstance(q, str) or not q.strip() or len(q) > settings.max_question_chars
        for q in value
    ):
        raise ClientFault(
            "invalid_questions",
            f"Each question must contain 1–{settings.max_question_chars} characters of text.",
        )
    return [q.strip() for q in value]


def document_kind(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in {".json", ".pdf"}:
        raise ClientFault("unsupported_type", "Document must be a .pdf or .json file.", 415)
    return extension[1:]


def parse_document(raw: bytes, kind: str, limits: dict) -> list[Chunk]:
    # Only non-secret parser settings cross the process boundary.
    settings = Settings(_env_file=None, **limits)
    if not raw:
        raise ClientFault("empty_document", "The document is empty.")
    if len(raw) > settings.max_document_bytes:
        raise ClientFault("document_too_large", "Document exceeds the upload size limit.", 413)
    if kind == "pdf":
        documents = _pdf_documents(raw, settings)
    else:
        documents = _json_documents(raw, settings)
    if not documents:
        raise ClientFault(
            "empty_document", "Document contains no usable text. Scans need OCR first."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=180)
    pieces = splitter.split_documents(documents)
    if len(pieces) > settings.max_chunks:
        raise ClientFault(
            "too_many_chunks", "Document is too large to index. Split it into parts.", 413
        )
    return [
        Chunk(f"c{i + 1:04d}", d.page_content, d.metadata["location"]) for i, d in enumerate(pieces)
    ]


def _pdf_documents(raw: bytes, settings: Settings) -> list[Document]:
    if not raw.startswith(b"%PDF-"):
        raise ClientFault("invalid_pdf", "The uploaded file is not a valid PDF.")
    # Parser diagnostics can contain fragments of untrusted file contents.
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ClientFault("encrypted_pdf", "Please upload a PDF without password protection.")
        if len(reader.pages) > settings.max_pages:
            raise ClientFault("too_many_pages", f"PDF limit is {settings.max_pages} pages.", 413)
        documents, count = [], 0
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            count += len(text)
            if count > settings.max_chars:
                raise ClientFault("too_much_text", "Extracted PDF text exceeds the limit.", 413)
            if text:
                documents.append(
                    Document(page_content=text, metadata={"location": f"page {number}"})
                )
        return documents
    except ClientFault:
        raise
    except Exception:
        raise ClientFault("invalid_pdf", "Could not read this PDF. Try a text-based PDF.") from None


def _json_documents(raw: bytes, settings: Settings) -> list[Document]:
    root = strict_json(raw)
    documents: list[Document] = []
    total = 0

    def visit(value, path: str, depth: int):
        nonlocal total
        if depth > 20:
            raise ClientFault("json_too_deep", "JSON nesting exceeds 20 levels.")
        if isinstance(value, dict):
            for key, child in value.items():
                token = key.replace("~", "~0").replace("/", "~1")
                visit(child, path + "/" + token, depth + 1)
        elif isinstance(value, list):
            # Keep a record together so a Q/A pair retains its relationship.
            for i, child in enumerate(value):
                if isinstance(child, dict) and all(
                    not isinstance(v, (dict, list)) for v in child.values()
                ):
                    add(json.dumps(child, ensure_ascii=False), path + f"/{i}")
                else:
                    visit(child, path + f"/{i}", depth + 1)
        elif value is not None and str(value).strip():
            add(f"{path or '/'}: {json.dumps(value, ensure_ascii=False)}", path or "/")

    def add(text: str, path: str):
        nonlocal total
        total += len(text)
        if total > settings.max_chars or len(documents) >= settings.max_chunks:
            raise ClientFault("too_much_text", "JSON content exceeds the indexing limit.", 413)
        documents.append(Document(page_content=text, metadata={"location": f"JSON pointer {path}"}))

    visit(root, "", 0)
    return documents
