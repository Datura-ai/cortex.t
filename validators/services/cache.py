import random
import sqlite3
import time
import hashlib
from typing import List

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

    def set_cache_in_batch(self, syns: List[StreamPrompting], ttl=3600 * 24, block_num=0, cycle_num=0, epoch_num=0):
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
            datas.append((p_key, syn.json(
                exclude={"dendrite", "completion", "total_size", "header_size", "axon", "uid", "provider", "model",
                         "required_hash_fields", "computed_body_hash", "streaming", "deserialize_flag", "task_id", }),
                          syn.completion, syn.provider, syn.model,
                          last_update_time))

        # Insert multiple records
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO cache (p_key, question, answer, provider, model, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', datas)

        # Commit the transaction
        self.conn.commit()

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
