"""Module for enums."""

from enum import StrEnum


class SentenceTransformerDevices(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
