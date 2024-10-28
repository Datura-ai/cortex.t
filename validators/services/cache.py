import random
import sqlite3
import time
import hashlib
from typing import List
import json
import requests
import bittensor as bt
from cortext import StreamPrompting


class QueryResponseCache:
    def __init__(self, validator_info=None):
        self.vali_hotkey = None
        self.vali_uid = None
        # Connect to (or create) the SQLite database
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()

        # Create a table for caching (key, value, and expiry time)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            p_key TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            provider TEXT,
            model TEXT,
            timestamp REAL
        )
        ''')
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_provider_model ON cache (provider, model);
        ''')
        conn.commit()
        self.conn = conn

    @staticmethod
    def generate_hash(input_string):
        return hashlib.sha256(input_string.encode('utf-8')).hexdigest()

    def set_vali_info(self, vali_hotkey, vali_uid):
        self.vali_hotkey = vali_hotkey
        self.vali_uid = vali_uid

    def set_cache(self, question, answer, provider, model, ttl=3600 * 24):
        return
        p_key = self.generate_hash(str(question) + str(provider) + str(model))
        expires_at = time.time() + ttl
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO cache (p_key, question, answer, provider, model, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (p_key, question, answer, provider, model, expires_at))
        self.conn.commit()

    def send_to_central_server(self, url, datas):
        start_time = time.time()
        if not url:
            return
        bt.logging.info("sending datas to central server.")
        headers = {
            'Content-Type': 'application/json'  # Specify that we're sending JSON
        }
        response = requests.post(url, data=json.dumps(datas), headers=headers)
        # Check the response
        if response.status_code == 200:
            bt.logging.info(
                f"Successfully sent data to central server. {time.time() - start_time} sec total elapsed for sending to central server.")
            return True
        else:
            bt.logging.info(
                f"Failed to send data. Status code: {response.status_code} {time.time() - start_time} sec total elapsed for sending to central server.")
            bt.logging.info(f"Response:{response.text}")
            return False

    def set_cache_in_batch(self, central_server_url, syns: List[StreamPrompting],
                           ttl=3600 * 24, block_num=0, cycle_num=0, epoch_num=0):
        datas = []
        last_update_time = time.time()
        for syn in syns:
            p_key = self.generate_hash(str(time.monotonic_ns()) + str(syn.json()) + str(random.random()))
            syn.time_taken = syn.dendrite.process_time or 0
            syn.validator_info = {"vali_uid": self.vali_uid, "vali_hotkey": self.vali_hotkey}
            syn.miner_info = {"miner_id": syn.uid, "miner_hotkey": syn.axon.hotkey}
            syn.block_num = block_num
            syn.epoch_num = epoch_num
            syn.cycle_num = cycle_num
            datas.append({"p_key": p_key,
                          "question": json.dumps(syn.json(
                              exclude={"dendrite", "completion", "total_size", "header_size", "axon", "uid", "provider",
                                       "model",
                                       "required_hash_fields", "computed_body_hash", "streaming", "deserialize_flag",
                                       "task_id", })),
                          "answer": syn.completion,
                          "provider": syn.provider,
                          "model": syn.model,
                          "timestamp": last_update_time})

        if self.send_to_central_server(central_server_url, datas):
            return
        # if not saved in central server successfully, then just save local cache.db file
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO cache (p_key, question, answer, provider, model, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [list(item.values()) for item in datas])

        # Commit the transaction
        self.conn.commit()
        return datas

    def get_answer(self, question, provider, model):
        p_key = self.generate_hash(str(question) + str(provider) + str(model))
        cursor = self.conn.cursor()
        cursor.execute('''
                SELECT answer FROM cache WHERE p_key = ?
                ''', (p_key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_cache(self, key):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT * FROM cache WHERE p_key = ?
        ''', key)
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_question_to_answers(self, provider, model):
        cursor = self.conn.cursor()
        cursor.execute('''
                SELECT question, answer FROM cache WHERE provider = ? AND model = ?
                ''', (provider, model))
        results = [(row[0], row[1]) for row in cursor.fetchall()]
        return results

    def close(self):
        self.conn.close()


cache_service = QueryResponseCache()
