import asyncio

import redis
import json
import bittensor as bt
from cortext import ALL_SYNAPSE_TYPE, StreamPrompting, ImageResponse


class Worker:
    # Initialize Redis client
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    TASK_STREAM = 'task_stream'
    RESULT_STREAM = 'result_stream'

    def __init__(self, worker_id, bandwidth, config, axon):
        self.worker_id = worker_id
        self.bandwidth = bandwidth
        self.dendrite = bt.dendrite(config.wallet)
        self.axon = axon
        self.report_resources()

    def report_resources(self):
        # Store worker's resource info in Redis hash
        self.redis_client.hset("workers", self.worker_id, self.bandwidth)

    @staticmethod
    def covert_json_to_synapse(task_obj):
        if task_obj.get("streaming"):
            synapse = StreamPrompting.parse_obj(task_obj)
        else:
            synapse = ImageResponse.parse_obj(task_obj)
        return synapse

    async def pull_and_run_task(self):
        # Pull task from worker-specific queue
        while True:
            task = json.loads(self.redis_client.lpop(f"tasks:{self.worker_id}") or "{}")
            if task:
                synapse = self.covert_json_to_synapse(task)
                bt.logging.trace(f"Worker {self.worker_id} received task: {synapse}")
                task_id = synapse.task_id
                try:
                    responses = await self.dendrite(
                        axons=[self.axon],
                        synapse=synapse,
                        deserialize=synapse.deserialize,
                        timeout=synapse.timeout,
                        streaming=synapse.streaming,
                    )
                except Exception as err:
                    bt.logging.exception(err)
                else:
                    async for chunk in responses[0]:
                        if isinstance(chunk, str):
                            await self.redis_client.xadd(Worker.RESULT_STREAM, {'task_id': task_id, 'chunk': chunk})
            await asyncio.sleep(0.1)
