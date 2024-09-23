import redis
import json
import bittensor as bt


class TaskMgr:
    def __init__(self):
        # Initialize Redis client
        self.redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        self.available_works = []
        self.workers = self.get_available_workers()

    def assign_task(self, task):

        # Find the worker with the most available resources (simplified logic)
        selected_worker = max(self.workers,
                              key=lambda w: self.workers[w])  # Example: Assign to worker with max remaining bandwidth
        if self.workers[selected_worker] <= 0:
            bt.logging.debug(f"no available resources to assign this task.")
            return None

        bt.logging.debug(f"Assigning task {task} to {selected_worker}")
        # decrease remaining capacity after sending request.
        self.workers[selected_worker] -= 1
        # Push task to the selected worker's task queue
        self.redis_client.rpush(f"tasks:{selected_worker}", task)

    def get_available_workers(self):
        # Get all workers' resource info
        workers = self.redis_client.hgetall("workers")
        worker_to_bandwidth = {}
        for worker_id, bandwidth in workers.items():
            worker_to_bandwidth[worker_id] = bandwidth
        return worker_to_bandwidth
