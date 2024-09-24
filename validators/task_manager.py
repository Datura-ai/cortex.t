import asyncio

import redis
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from miner.config import config
from validators.workers import Worker
from validators import utils


class TaskMgr:
    def __init__(self, uid_to_capacities, config):
        # Initialize Redis client
        self.redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        self.resources = []
        self.create_workers(uid_to_capacities)
        self.config = config

    def assign_task(self, synapse: ALL_SYNAPSE_TYPE):

        # Find the worker with the most available resources (simplified logic)
        selected_worker = max(self.resources,
                              key=lambda w: self.resources[w])  # Example: Assign to worker with max remaining bandwidth
        if self.resources[selected_worker] <= 0:
            bt.logging.debug(f"no available resources to assign this task.")
            return None

        task_id = utils.create_hash_value((synapse.json()))
        bt.logging.trace(f"Assigning task {task_id} to {selected_worker}")

        # decrease remaining capacity after sending request.
        self.resources[selected_worker] -= 1
        # Push task to the selected worker's task queue
        worker = Worker(worker_id="123", config=config, axon=selected_worker.axon)
        self.redis_client.rpush(f"tasks:{task_id}", synapse.json())
        asyncio.create_task(worker.pull_and_run_task())

    def create_workers(self, uid_to_capacities):
        # create worker for each uid, provider, model
        workers = []
        for uid, cap_info in uid_to_capacities.items():
            for provider, model_to_cap in cap_info.items():
                for model, cap in model_to_cap.items():
                    worker_id = f"{uid}_{provider}_{model}"
                    worker = Worker(worker_id, cap, config=self.config)
                    workers.append(worker)
        self.resources = workers
