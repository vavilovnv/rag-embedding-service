"""Module with utils."""

from app.settings import settings


def get_collection_name(source: str | None) -> str:
    """Generate collection name from source."""
    return source if source else settings.EMBEDDING_COLLECTION_DEFAULT_NAME
