import io
import json

import pytest
from pypdf import PdfReader, PdfWriter

from app.ingestion import document_kind, parse_document, parse_questions, strict_json
from app.models import ClientFault
from tests.conftest import pdf_bytes


def test_pdf_text_and_page_provenance():
    chunks = parse_document(pdf_bytes(pages=2), "pdf", {})
    assert {c.location for c in chunks} == {"page 1", "page 2"}
    assert "AWS" in chunks[0].text


def test_json_records_preserve_relationship_and_pointer():
    raw = json.dumps({"a/b": [{"question": "Provider?", "answer": "AWS"}]}).encode()
    chunks = parse_document(raw, "json", {})
    assert len(chunks) == 1
    assert "Provider?" in chunks[0].text and "AWS" in chunks[0].text
    assert chunks[0].location == "JSON pointer /a~1b/0"


@pytest.mark.parametrize("raw", [b"{broken", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"])
def test_invalid_json(raw):
    with pytest.raises(ClientFault, match="invalid_json"):
        strict_json(raw)


@pytest.mark.parametrize(
    "value", [[], [""], [" "], [4], {"questions": ["x"], "extra": 1}, ["x"] * 21]
)
def test_invalid_questions(value, settings):
    with pytest.raises(ClientFault):
        parse_questions(json.dumps(value).encode(), settings)


def test_question_wrapper_and_duplicates(settings):
    assert parse_questions(b'{"questions":[" A? ","A?"]}', settings) == ["A?", "A?"]


def test_question_length_and_bytes(settings):
    with pytest.raises(ClientFault):
        parse_questions(json.dumps(["x" * 1001]).encode(), settings)
    with pytest.raises(ClientFault) as error:
        parse_questions(b"x" * 65537, settings)
    assert error.value.status == 413


@pytest.mark.parametrize(
    "raw,kind",
    [
        (b"", "json"),
        (b"{}", "json"),
        (b"[]", "json"),
        (b"null", "json"),
        (pdf_bytes(text=""), "pdf"),
    ],
)
def test_empty_documents(raw, kind):
    with pytest.raises(ClientFault, match="empty_document"):
        parse_document(raw, kind, {})


@pytest.mark.parametrize("raw", [b"not a pdf", b"%PDF-1.7\ngarbage"])
def test_invalid_pdf(raw):
    with pytest.raises(ClientFault, match="invalid_pdf"):
        parse_document(raw, "pdf", {})


def test_encrypted_pdf():
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes())))
    writer.encrypt("secret")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ClientFault, match="encrypted_pdf"):
        parse_document(output.getvalue(), "pdf", {})


def test_page_size_text_and_chunk_limits():
    with pytest.raises(ClientFault, match="too_many_pages"):
        parse_document(pdf_bytes(pages=2), "pdf", {"max_pages": 1})
    with pytest.raises(ClientFault, match="document_too_large"):
        parse_document(b"x" * 101, "json", {"max_document_bytes": 100})
    with pytest.raises(ClientFault, match="too_much_text"):
        parse_document(json.dumps({"body": "x" * 200}).encode(), "json", {"max_chars": 100})
    with pytest.raises(ClientFault, match="too_many_chunks"):
        parse_document(
            json.dumps({"body": "some facts " * 1000}).encode(), "json", {"max_chunks": 1}
        )


def test_nesting_limit():
    value = "text"
    for _ in range(22):
        value = {"child": value}
    with pytest.raises(ClientFault, match="json_too_deep"):
        parse_document(json.dumps(value).encode(), "json", {})


def test_type_allowlist():
    assert document_kind("REPORT.PDF") == "pdf"
    with pytest.raises(ClientFault) as error:
        document_kind("script.html")
    assert error.value.status == 415


@pytest.mark.parametrize("password", ["", "required-secret"])
def test_aes_pdf_open_password_handling(password):
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes(pages=2))))
    writer.encrypt(password, owner_password="owner-secret", algorithm="AES-256")
    output = io.BytesIO()
    writer.write(output)
    if password:
        with pytest.raises(ClientFault, match="encrypted_pdf"):
            parse_document(output.getvalue(), "pdf", {})
    else:
        chunks = parse_document(output.getvalue(), "pdf", {})
        assert {c.location for c in chunks} == {"page 1", "page 2"}
        assert all("AWS" in c.text for c in chunks)
        with pytest.raises(ClientFault, match="too_many_pages"):
            parse_document(output.getvalue(), "pdf", {"max_pages": 1})


def test_missing_crypto_dependency_is_not_mislabeled_invalid_pdf(monkeypatch):
    from pypdf.errors import DependencyError

    def fail(*args, **kwargs):
        raise DependencyError("internal dependency details")

    monkeypatch.setattr("app.ingestion.PdfReader", fail)
    with pytest.raises(ClientFault) as exc:
        parse_document(pdf_bytes(), "pdf", {})
    assert exc.value.code == "pdf_dependency_missing" and exc.value.status == 503
    assert "internal" not in exc.value.message
