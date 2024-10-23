from fastapi import FastAPI, Depends, HTTPException
from . import curd, models, schemas
from .database import create_table

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    create_table(None)


# Create an item
@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate):
    return curd.create_item(item=item)


# Read all items
@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 10):
    items = curd.get_items(skip=skip, limit=limit)
    return items


# Read a single item by ID
@app.get("/items/{p_key}", response_model=schemas.Item)
def read_item(p_key: int):
    db_item = curd.get_item(p_key=p_key)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
