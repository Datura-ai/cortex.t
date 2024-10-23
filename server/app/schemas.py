from pydantic import BaseModel, Json


class ItemBase(BaseModel):
    p_key: str
    question: Json
    answer: str
    provider: str
    model: str
    timestamp: float


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    pass
