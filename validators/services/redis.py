import aioredis
import asyncio
import bittensor as bt
from cortext import REDIS_RESULT_STREAM


class Redis:
    def __init__(self):
        pass

    @staticmethod
    async def get_stream_result(redis_client, task_id):
        last_id = '0'  # Start reading from the beginning of the stream
        bt.logging.trace(f"Waiting for results of task {task_id}...")
        stream_name = REDIS_RESULT_STREAM + f"{task_id}"

        while True:
            # Read from the Redis stream
            result_entries = redis_client.xread({stream_name: last_id}, block=5000)
            result_entries = result_entries or []

            for entry in result_entries:
                stream_name, results = entry
                for result_id, data in results:
                    result_task_id = data[b'task_id'].decode()
                    result_chunk = data[b'chunk'].decode()
                    # Only process results for the specific task
                    if result_task_id == task_id:
                        yield result_chunk
            else:
                bt.logging.trace("No new results, waiting...")
                break
        bt.logging.trace(f"stream exit. delete old messages from queue.")
        await redis_client.xtrim(stream_name, maxlen=0, approximate=False)

    def get_result(self, task_id):
        pass
