import redis
import time
import bittensor as bt


class Worker:
    # Initialize Redis client
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    def __init__(self, worker_id, bandwidth, config):
        self.worker_id = worker_id
        self.bandwidth = bandwidth
        self.dendrite = bt.dendrite(config.wallet)
        self.report_resources()

    def report_resources(self):
        # Store worker's resource info in Redis hash
        self.redis_client.hset("workers", self.worker_id, self.bandwidth)

    async def pull_task(self):
        # Pull task from worker-specific queue
        task = self.redis_client.lpop(f"tasks:{self.worker_id}")
        if task:
            bt.logging.trace(f"Worker {self.worker_id} received task: {task}")
            # Execute the task (in this example, just print the task)
            try:
                result = await self.dendrite.query(axons=[task.axon], synapse=task.synapse)
            except Exception as err:
                bt.logging.exception(err)
            bt.logging.trace(f"Worker {self.worker_id} completed task: {task}")
            return result
        return False
