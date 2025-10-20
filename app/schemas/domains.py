from pydantic import BaseModel


class Texts(BaseModel):
    texts: list[str]
