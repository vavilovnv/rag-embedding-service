from fastapi import FastAPI

from app.schemas.domains import Texts
from app.services.embedding_model import model

app = FastAPI(title="Embedding Service")


@app.post("/embed")
def embed_texts(data: Texts) -> dict[str, list[list[float]]]:
    embeddings = model.encode(data.texts, convert_to_numpy=True).tolist()
    return {"embeddings": embeddings}
