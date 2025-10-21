"""Module for RAG content methods."""

import asyncio

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.resources import strings
from app.services.embeddings import EmbeddingStore
from app.settings import settings


async def get_chroma_content(source: str) -> list[dict]:
    """Get content pages from vector DB."""
    vector_store = await EmbeddingStore.get_store(source)
    try:
        collection = vector_store._collection  # noqa
    except Exception as exc:
        return [
            {"error": strings.VDB_GETTING_DATA_ERROR_TEMPLATE.format(error=str(exc))}
        ]

    return await asyncio.to_thread(_process_collection_data, collection)


async def update_rag_embeddings(source: str, text: str) -> int:
    """Update embeddings for RAG in vector DB."""
    docs = [Document(page_content=text, metadata={"source": source})]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ".\n"],
        chunk_size=settings.EMBEDDING_PAGE_CHUNK_SIZE,
        chunk_overlap=settings.EMBEDDING_PAGE_CHUNK_OVERLAP,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)

    vector_store = await EmbeddingStore.get_store(source)
    ids = await vector_store.aadd_documents(splits)

    return len(ids)


async def search_for_message(source: str, user_message: str) -> str:
    """Search text chunks for users' message."""
    vector_store = await EmbeddingStore.get_store(source)

    return await asyncio.to_thread(_do_similarity_search, vector_store, user_message)


def _process_collection_data(collection: chromadb.Collection) -> list[dict]:
    documents: chromadb.GetResult = collection.get()

    content_list = []
    if documents and "documents" in documents:
        base_data = zip(documents["documents"], documents["metadatas"], strict=True)  # type: ignore
        for obj_id, (doc, metadata) in enumerate(base_data):
            content_list.append(
                {
                    "id": obj_id,
                    "content": doc[:97] + "..." if len(doc) >= 100 else doc,
                    "metadata": metadata,
                }
            )

    return content_list


def _do_similarity_search(vector_store: Chroma, user_message: str) -> str:
    rag_data = vector_store.similarity_search(
        user_message, k=settings.SEARCH_RESULTS_NUMBER
    )
    return "\n".join([doc.page_content for doc in rag_data])
