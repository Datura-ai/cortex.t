import sqlite3
import time
import hashlib


class QueryResponseCache:
    def __init__(self):
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

    def set_cache(self, question, answer, provider, model, ttl=3600*24):
        p_key = self.generate_hash(str(question) + str(provider) + str(model))
        expires_at = time.time() + ttl
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO cache (p_key, question, answer, provider, model, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (p_key, question, answer, provider, model, expires_at))
        self.conn.commit()

    def get_cache(self, key):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT value FROM cache WHERE p_key = ? AND timestamp > ?
        ''', (key, time.time()))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_question_to_answers(self, provider, model):
        cursor = self.conn.cursor()
        cursor.execute('''
                SELECT value FROM cache WHERE provider = ? AND model = ?
                ''', (provider, model))
        result = cursor.fetchall()
        return list(result) if result else None

    def close(self):
        self.conn.close()


cache_service = QueryResponseCache()