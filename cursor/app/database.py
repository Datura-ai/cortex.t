import psycopg2
import os

DATABASE_URL = os.getenv("DATABASE_URL")
TABEL_NAME = 'query_resp_data'
# PostgreSQL connection parameters
conn = psycopg2.connect(DATABASE_URL)

# Create a cursor object to interact with the database
cur = conn.cursor()


async def create_table(app):
    global conn, cur, TABEL_NAME
    try:
        pass

    except Exception as e:
        print(f"Error creating table: {e}")

create_table(None)
