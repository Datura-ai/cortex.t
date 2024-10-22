from sqlalchemy import Column, Integer, JSON, String
from .database import Base

class Item(Base):
    __tablename__ = "query_resp_score"

    id = Column(String, primary_key=True, index=True)
    query = Column(JSON)  # JSON column
    answer = Column(String)
    provider = Column(String)
    model = Column(String)
    time_stamp = Column(Integer)