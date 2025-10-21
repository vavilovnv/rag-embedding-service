"""Common endpoints."""

from fastapi import APIRouter, status

router = APIRouter(
    tags=["common"],
)


@router.get("/", status_code=status.HTTP_400_BAD_REQUEST)
async def root():
    return {"message": "Not available"}


@router.get("/healthcheck", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "ok"}
