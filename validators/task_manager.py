import redis
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.workers import Worker


class TaskMgr:
    def __init__(self, uid_to_capacities, config):
        # Initialize Redis client
        self.redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        self.workers = []
        self.create_workers(uid_to_capacities)
        self.config = config

    def assign_task(self, task: ALL_SYNAPSE_TYPE):

        # Find the worker with the most available resources (simplified logic)
        selected_worker = max(self.workers,
                              key=lambda w: self.workers[w])  # Example: Assign to worker with max remaining bandwidth
        if self.workers[selected_worker] <= 0:
            bt.logging.debug(f"no available resources to assign this task.")
            return None

        bt.logging.trace(f"Assigning task {task} to {selected_worker}")
        # decrease remaining capacity after sending request.
        self.workers[selected_worker] -= 1
        # Push task to the selected worker's task queue
        self.redis_client.rpush(f"tasks:{selected_worker}", task.json())

    def create_workers(self, uid_to_capacities):
        # create worker for each uid, provider, model
        workers = []
        for uid, cap_info in uid_to_capacities.items():
            for provider, model_to_cap in cap_info.items():
                for model, cap in model_to_cap.items():
                    worker_id = f"{uid}_{provider}_{model}"
                    worker = Worker(worker_id, cap, config=self.config)
                    workers.append(worker)
        self.workers = workers
