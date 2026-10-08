from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    location: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    quote: str


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "not_found"]
    answer: str
    evidence: list[Evidence]


class Citation(BaseModel):
    chunk_id: str
    location: str
    quote: str


class Answer(BaseModel):
    question: str
    status: Literal["answered", "not_found", "error"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    error_code: str | None = None


class Usage(BaseModel):
    embedding_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0


class QAResponse(BaseModel):
    request_id: str
    results: list[Answer]
    usage: Usage
    duration_ms: int
    document_chunks: int


class ClientFault(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        self.code, self.message, self.status = code, message, status
        # Preserve constructor arguments across parser subprocess pickling.
        super().__init__(code, message, status)


class ProviderFault(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)
