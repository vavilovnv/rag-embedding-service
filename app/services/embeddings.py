import asyncio
import hashlib

import chromadb
from chromadb import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.resources import strings
from app.settings import settings


class E5Embeddings(Embeddings):
    """Class for embeddings using in vector DB."""

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(
            model_path, device=settings.EMBEDDING_PROCESSING_DEVICE
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts_with_prefix = [f"passage: {text}" for text in texts]
        return self.model.encode(texts_with_prefix).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([f"query: {text}"])[0].tolist()


class EmbeddingStore:
    """Class for work with vector DB."""

    _lock = asyncio.Lock()
    _client: chromadb.PersistentClient | None = None
    _embeddings: E5Embeddings | None = None
    _stores: dict[str, Chroma] = {}

    @classmethod
    async def get_store(cls, source: str = "default-store") -> Chroma:
        """Get or create a store for the specific source collection."""
        collection_name = cls._get_chroma_db_collection_name(source)

        if collection_name in cls._stores:
            return cls._stores[collection_name]

        async with cls._lock:
            if collection_name in cls._stores:
                return cls._stores[collection_name]

            if cls._client is None:
                await asyncio.to_thread(cls._init_client)

            store = await asyncio.to_thread(cls._create_store, collection_name)
            cls._stores[collection_name] = store

            return store

    @classmethod
    async def clear_cache(cls, source: str | None = None) -> None:
        """Clear cached store for the specific source or all stores."""
        async with cls._lock:
            if source:
                collection_name = cls._get_chroma_db_collection_name(source)
                cls._stores.pop(collection_name, None)
            else:
                cls._stores.clear()

    @classmethod
    async def delete_collection(cls, source: str) -> None:
        """Delete collection data from ChromaDB and clear cache."""
        if not source:
            return

        async with cls._lock:
            collection_name = cls._get_chroma_db_collection_name(source)

            cls._stores.pop(collection_name, None)

            if cls._client is not None:
                await asyncio.to_thread(
                    cls._client.delete_collection, name=collection_name
                )

    @classmethod
    def _init_client(cls) -> None:
        """Initialize ChromaDB client and embeddings model once."""
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=settings.EMBEDDING_DB_PATH,
                settings=Settings(anonymized_telemetry=False),
            )

        if cls._embeddings is None:
            if not settings.EMBEDDING_MODEL_PATH:
                raise ValueError("Embedding model path is not set")

            cls._embeddings = E5Embeddings(
                model_path=settings.EMBEDDING_MODEL_PATH,
            )

    @classmethod
    def _create_store(cls, collection_name: str) -> Chroma:
        """Create a Chroma store for the specific collection."""
        if cls._client is None or cls._embeddings is None:
            raise RuntimeError("Client not initialized")

        return Chroma(
            client=cls._client,
            collection_name=collection_name,
            embedding_function=cls._embeddings,
        )

    @staticmethod
    def _get_chroma_db_collection_name(source: str) -> str:
        """Generate collection name from source."""
        source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
        return strings.VDB_COLLECTION_NAME_TEMPLATE.format(url_hash=source_hash[:15])


__all__ = ["EmbeddingStore"]
