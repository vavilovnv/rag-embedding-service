"""Embeddings endpoints."""

from fastapi import APIRouter, status

from app.schemas.domains import Source, SourceContent
from app.services.rag import get_chroma_content, update_rag_embeddings
from app.services.utils import get_collection_name

router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"],
)


@router.get("/get_collection", status_code=status.HTTP_200_OK)
async def get_embeddings(payload: Source) -> dict[str, list[dict]]:
    embeddings = await get_chroma_content(source=get_collection_name(payload.source))
    return {"embeddings": embeddings}


@router.post("/update", status_code=status.HTTP_200_OK)
async def update_embeddings(payload: SourceContent) -> dict[str, int]:
    pages_count = await update_rag_embeddings(
        source=get_collection_name(payload.source), text=payload.text
    )
    return {"pages": pages_count}
