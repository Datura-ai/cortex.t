import psycopg2
import os
from contextlib import asynccontextmanager

DATABASE_URL = os.getenv("DATABASE_URL")
TABEL_NAME = 'query_resp_data'
# PostgreSQL connection parameters
conn = psycopg2.connect(DATABASE_URL)

# Create a cursor object to interact with the database
cur = conn.cursor()


async def create_table(app):
    global conn, cur, TABEL_NAME
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # SQL command to create a table
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABEL_NAME} (
            p_key VARCHAR(100) PRIMARY KEY,
            question JSON,
            answer TEXT,
            provider VARCHAR(100),
            model VARCHAR(100),
            timestamp FLOAT,
            miner_hot_key VARCHAR(100),
            miner_uid INTEGER,
            score FLOAT,
            similarity FLOAT,
            vali_uid INTEGER,
            timeout INTEGER,
            time_taken INTEGER,
            epoch_num INTEGER,
            cycle_num INTEGER,
            block_num INTEGER,
            name VARCHAR(100)
        );
        """

        # Execute the SQL command
        cur.execute(create_table_query)
        conn.commit()  # Save changes
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS question_answer_index ON {TABEL_NAME} (provider, model);
        """
        cur.execute(create_index_query)
        conn.commit()
        print("Table created successfully!")

    except Exception as e:
        print(f"Error creating table: {e}")

create_table(None)
