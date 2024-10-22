from pydantic import BaseModel, Json
from typing import Any, Dict

class ItemBase(BaseModel):
    id: int
    question: Json
    answer: str
    provider: str
    model: str
    time_stamp: int

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True