from typing import Union, Optional
from pydantic import BaseModel, Field


class Filters(BaseModel):
    min_score: float = Field(0, ge=0)
    min_similarity: float = Field(0.0, ge=0)
    model: Optional[str] = Field("")
    provider: Optional[str] = Field("")
    epoch_num: Optional[int] = Field("")
    cycle_num: Optional[int] = Field("")
    block_num: Optional[int] = Field("")
    name: Optional[str] = Field("")
    min_timestamp: Optional[float] = Field(0, ge=0)
    max_timestamp: Optional[float] = Field(999999999999, ge=0)



class RequestBody(BaseModel):
    filters: Optional[Filters] = Field(..., description="filter for searching")
    search: Optional[Union[int, str]] = Field(..., description="An integer ID or a string search key")
    sort_by: Optional[str] = Field("miner_uid", description="Field to sort by")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Sorting order, 'asc' or 'desc'")
    skip: Optional[int] = Field(0, description="skip for pagination")
    limit: Optional[int] = Field(100, description="skip for pagination")
