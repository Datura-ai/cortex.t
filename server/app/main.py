from fastapi import FastAPI, Depends, HTTPException
from . import crud, models, schemas
from .database import create_table

app = FastAPI()
app.add_event_handler("lifespan", create_table)


# Create an item
@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate):
    return crud.create_item(item=item)


# Read all items
@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 10):
    items = crud.get_items(skip=skip, limit=limit)
    return items


# Read a single item by ID
@app.get("/items/{item_id}", response_model=schemas.Item)
def read_item(item_id: int):
    db_item = crud.get_item(item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
