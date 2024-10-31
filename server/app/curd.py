import traceback

import psycopg2
import os
from typing import List
from . import models, schemas
from .database import cur, TABEL_NAME, conn, DATABASE_URL
from fastapi import HTTPException


def create_item(item: schemas.ItemCreate):
    global cur, conn
    query = f"INSERT INTO {TABEL_NAME} (p_key, question, answer, provider, model, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"
    cur.execute(query, item.p_key, item.question, item.answer, item.provider, item.model, item.timestamp)
    conn.commit()  # Save changes to the database
    return item


def create_items(items: List[schemas.ItemCreate]):
    conn = psycopg2.connect(DATABASE_URL)
    # Create a cursor object to interact with the database
    cur = conn.cursor()
    query = f"INSERT INTO {TABEL_NAME} (p_key, question, answer, provider, model, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"
    datas = []
    for item in items:
        miner_hot_key = item.question.get("miner_info", {}).get("miner_id")
        miner_uid = item.question.get("miner_info", {}).get("miner_hotkey")
        score = item.question.get("score")
        similarity = item.question.get("similarity")
        vali_uid = item.question.get("validator_info").get("vali_uid")
        timeout = item.question.get("timeout")
        time_taken = item.question.get("time_taken")
        epoch_num = item.question.get("epoch_num")
        cycle_num = item.question.get("cycle_num")
        block_num = item.question.get("block_num")
        name = item.question.get("name")
        datas.append((item.p_key, item.question, item.answer, item.provider, item.model, item.timestamp, miner_hot_key, miner_uid,
                      score, similarity, vali_uid, timeout, time_taken, epoch_num, cycle_num, block_num, name))
    try:
        if conn.closed:
            print("connection is closed already")
        cur.executemany(query, datas)
        conn.commit()  # Save changes to the database
        print("successfully saved in database")
    except Exception as err:
        print(err, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error {err}")


def get_items(req_body: models.RequestBody):
    conn = psycopg2.connect(DATABASE_URL)
    # Create a cursor object to interact with the database
    cur = conn.cursor()
    skip = req_body.skip
    limit = req_body.limit
    filter_by_miner_score = f"score>={req_body.filters.min_score}"
    filter_by_miner_similarity = f"score>={req_body.filters.min_similarity}"
    filter_by_provider = f"provider={req_body.filters.provider}"
    filter_by_model = f"model={req_body.filters.model}"
    filter_by_min_timestamp = f"timestamp>{req_body.filters.min_timestamp}"
    filter_by_max_timestamp = f"timestamp<={req_body.filters.max_timestamp}"
    query = f"SELECT * FROM {TABEL_NAME} offset {skip} limit {limit} ;"
    cur.execute(query)
    items = cur.fetchall()  # Fetch all results
    return [item for item in items]


def get_item(p_key: int):
    query = f"SELECT * FROM {TABEL_NAME} WHERE p_key = %s"
    cur.execute(query, (p_key,))
    item = cur.fetchone()  # Fetch one result
    return dict(item)
