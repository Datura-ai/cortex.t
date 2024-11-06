import json
import time
import traceback

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

from . import models, schemas
from .database import TABEL_NAME, DATABASE_URL
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
    query = (f"INSERT INTO {TABEL_NAME} (p_key, question, answer, provider, model, timestamp, miner_hot_key, miner_uid"
             f", score, similarity, vali_uid, timeout, time_taken, epoch_num, cycle_num, block_num"
             f", name) VALUES (%s, %s, %s, %s, %s, %s"
             f", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
    datas = []
    for item in items:
        question = json.loads(item.question)
        miner_uid = question.get("miner_info", {}).get("miner_id") or 99999
        miner_hot_key = question.get("miner_info", {}).get("miner_hotkey") or ""
        score = question.get("score") or 0
        similarity = question.get("similarity") or 0
        vali_uid = question.get("validator_info").get("vali_uid")
        timeout = question.get("timeout")
        time_taken = question.get("time_taken")
        epoch_num = question.get("epoch_num")
        cycle_num = question.get("cycle_num")
        block_num = question.get("block_num")
        name = question.get("name") or ""
        datas.append((item.p_key, item.question, item.answer, item.provider, item.model, item.timestamp, miner_hot_key,
                      miner_uid, score, similarity, vali_uid, timeout,
                      time_taken, epoch_num, cycle_num, block_num, name))
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
    cur = conn.cursor(cursor_factory=RealDictCursor)
    skip = req_body.skip
    limit = req_body.limit

    filter_by_miner_score = f"score>={req_body.filters.min_score}" if req_body.filters.min_score else ""
    filter_by_miner_similarity = f"score>={req_body.filters.min_similarity}" if req_body.filters.min_similarity else ""
    filter_by_provider = f"provider='{req_body.filters.provider}'" if req_body.filters.provider else ""
    filter_by_model = f"model='{req_body.filters.model}'" if req_body.filters.model else ""
    filter_by_min_timestamp = f"timestamp>={req_body.filters.min_timestamp}" if req_body.filters.min_timestamp else ""
    filter_by_max_timestamp = f"timestamp<={req_body.filters.max_timestamp}" if req_body.filters.max_timestamp else ""
    filter_by_epoch_num = f"epoch_num={req_body.filters.epoch_num}" if req_body.filters.epoch_num else ""
    filter_by_block_num = f"block_num={req_body.filters.block_num}" if req_body.filters.block_num else ""
    filter_by_cycle_num = f"cycle_num={req_body.filters.cycle_num}" if req_body.filters.cycle_num else ""
    filter_by_name = f"name={req_body.filters.name}" if req_body.filters.name else ""
    search_by_uid_or_hotkey = (f"miner_uid=%s" if str(req_body.search).isdigit()
                               else f"miner_hot_key like %s") if req_body.search else ""
    conditions = [filter_by_miner_score, filter_by_miner_similarity, filter_by_provider, filter_by_model,
                  filter_by_min_timestamp,
                  filter_by_max_timestamp, filter_by_epoch_num, filter_by_block_num, filter_by_cycle_num,
                  filter_by_name, search_by_uid_or_hotkey]
    conditions = [item for item in conditions if item]
    conditions_query = " and ".join(conditions)
    order_by = f"order by {req_body.sort_by} {req_body.sort_order}"
    query = f"SELECT * FROM {TABEL_NAME} where {conditions_query} {order_by} limit {limit} offset {skip};"
    print(query)
    query_cnt = f"SELECT count(*) FROM {TABEL_NAME} where {conditions_query}"
    start_time = time.time()
    cur.execute(query, (f"%{req_body.search}%" if not str(req_body.search).isdigit() else str(req_body.search),))
    print(f"execution query has been completed. {time.time() - start_time}")
    items = cur.fetchall()  # Fetch all results
    print(f"loaded to python object array. {time.time() - start_time}")
    # cur.execute(query_cnt, (f"%{req_body.search}%" if not str(req_body.search).isdigit() else str(req_body.search),))
    # cnt = cur.fetchone().get('count')
    # print(f"all query execution has been succeed. {time.time() - start_time}")
    cur.close()
    conn.close()
    return {"records": items, "limit": limit, "skip": skip}


def get_item(p_key: int):
    query = f"SELECT * FROM {TABEL_NAME} WHERE p_key = %s"
    cur.execute(query, (p_key,))
    item = cur.fetchone()  # Fetch one result
    return dict(item)
