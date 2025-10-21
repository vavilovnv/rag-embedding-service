"""Models for domain schemas."""

from pydantic import BaseModel

from app.settings import settings


class Source(BaseModel):
    source: str | None = settings.EMBEDDING_COLLECTION_DEFAULT_NAME


class SourceContent(Source):
    text: str


class SourceQuestion(Source):
    question: str
