from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", extra="ignore")
    openai_api_key: SecretStr = SecretStr("")
    app_api_token: SecretStr = SecretStr("")
    # Model is deliberately fixed to the challenge requirement.
    model: str = Field(default="gpt-4o-mini", pattern=r"^gpt-4o-mini$")
    max_body_bytes: int = Field(default=11 * 1024 * 1024, ge=1024)
    max_document_bytes: int = Field(default=10 * 1024 * 1024, ge=100)
    max_questions_bytes: int = Field(default=64 * 1024, ge=100)
    max_questions: int = Field(default=20, ge=1, le=50)
    max_question_chars: int = Field(default=1000, ge=10)
    max_pages: int = Field(default=100, ge=1)
    max_chars: int = Field(default=250_000, ge=100)
    max_chunks: int = Field(default=300, ge=1)
    max_embedding_tokens: int = Field(default=80_000, ge=100)
    top_k: int = Field(default=6, ge=1, le=10)
    max_concurrent_requests: int = Field(default=2, ge=1, le=8)
    llm_concurrency: int = Field(default=4, ge=1, le=8)
    parser_timeout: float = Field(default=15, gt=0)
    answer_timeout: float = Field(default=30, gt=0)
    request_timeout: float = Field(default=120, gt=0)
    upload_timeout: float = Field(default=20, gt=0)
    max_full_context_questions: int = Field(default=75, ge=1, le=100)
    full_context_batch_size: int = Field(default=5, ge=1, le=10)
    full_context_input_tokens: int = Field(default=100_000, ge=1000, le=110_000)
    full_context_request_timeout: float = Field(default=300, gt=0)
    full_context_call_timeout: float = Field(default=60, gt=0)
