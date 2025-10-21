"""Main module."""

from fastapi import FastAPI

from app.api import common, content, embeddings
from app.settings import settings

app = FastAPI(
    title="Embedding Service",
    redoc_url=None,
    openapi_url=(None, "/openapi.json")[settings.DEBUG],
)

api_prefix = "/api/v1"

app.include_router(common.router)
app.include_router(embeddings.router, prefix=api_prefix)
app.include_router(content.router, prefix=api_prefix)
