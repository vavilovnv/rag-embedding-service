"""Content endpoints."""

from fastapi import APIRouter, status

from app.schemas.domains import SourceQuestion
from app.services.rag import search_for_message
from app.services.utils import get_collection_name

router = APIRouter(
    prefix="/content",
    tags=["content"],
)


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_content(payload: SourceQuestion) -> dict[str, str]:
    content = await search_for_message(
        source=get_collection_name(payload.source), user_message=payload.question
    )
    return {"content": content}
