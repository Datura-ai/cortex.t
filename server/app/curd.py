import os

from . import models, schemas
from database import cur

db_name = os.getenv('POSTGRES_DB')


def create_item(*args):
    query = f"INSERT INTO {db_name} (p_key, question, answer, provider, model, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"
    cur.execute(query, args)
    cur.commit()  # Save changes to the database
    print("item created successfully!")
    return args


def get_items(skip: int = 0, limit: int = 10):
    query = f"SELECT * FROM {db_name} LIMIT {limit} OFFSET {skip};"
    cur.execute(query)
    items = cur.fetchall()  # Fetch all results
    return [dict(item) for item in items]


def get_item(p_key: int):
    query = "SELECT * FROM users WHERE p_key = %s"
    cur.execute(query, (p_key,))
    item = cur.fetchone()  # Fetch one result
    return dict(item)
