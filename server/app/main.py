from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import engine, get_db

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Create an item
@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    return crud.create_item(db=db, item=item)

# Read all items
@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items

# Read a single item by ID
@app.get("/items/{item_id}", response_model=schemas.Item)
def read_item(item_id: int, db: Session = Depends(get_db)):
    db_item = crud.get_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

# Update an item
# @app.put("/items/{item_id}", response_model=schemas.Item)
# def update_item(item_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)):
#     updated_item = crud.update_item(db=db, item_id=item_id, item=item)
#     if updated_item is None:
#         raise HTTPException(status_code=404, detail="Item not found")
#     return updated_item
#
# # Delete an item
# @app.delete("/items/{item_id}", response_model=schemas.Item)
# def delete_item(item_id: int, db: Session = Depends(get_db)):
#     deleted_item = crud.delete_item(db=db, item_id=item_id)
#     if deleted_item is None:
#         raise HTTPException(status_code=404, detail="Item not found")
#     return deleted_item