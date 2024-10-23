import os
from typing import List
from . import models, schemas
from .database import cur, TABEL_NAME
from fastapi import HTTPException


def create_item(item: schemas.ItemCreate):
    query = f"INSERT INTO {TABEL_NAME} (p_key, question, answer, provider, model, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"
    cur.execute(query, item.p_key, item.question, item.answer, item.provider, item.model, item.timestamp)
    cur.commit()  # Save changes to the database
    return item


def create_items(items: List[schemas.ItemCreate]):
    query = f"INSERT INTO {TABEL_NAME} (p_key, question, answer, provider, model, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"
    datas = []
    for item in items:
        datas.append((item.p_key, item.question, item.answer, item.provider, item.model, item.timestamp))
    try:
        cur.executemany(query, datas)
        cur.commit()  # Save changes to the database
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Server Error {err}")


def get_items(skip: int = 0, limit: int = 10):
    query = f"SELECT * FROM {TABEL_NAME} LIMIT {limit} OFFSET {skip};"
    cur.execute(query)
    items = cur.fetchall()  # Fetch all results
    return [dict(item) for item in items]


def get_item(p_key: int):
    query = f"SELECT * FROM {TABEL_NAME} WHERE p_key = %s"
    cur.execute(query, (p_key,))
    item = cur.fetchone()  # Fetch one result
    return dict(item)
