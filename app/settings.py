"""Project settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.enums import SentenceTransformerDevices


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    EMBEDDING_PROCESSING_DEVICE: SentenceTransformerDevices = (
        SentenceTransformerDevices.CPU
    )
    EMBEDDING_MODEL_PATH: str = ""
    EMBEDDING_DB_PATH: str = ""
    EMBEDDING_PAGE_CHUNK_SIZE: int = 600
    EMBEDDING_PAGE_CHUNK_OVERLAP: int = 200
    EMBEDDING_COLLECTION_DEFAULT_NAME: str = "default-store"

    SEARCH_RESULTS_NUMBER: int = 5

    DEBUG: bool = False


settings = AppSettings()
