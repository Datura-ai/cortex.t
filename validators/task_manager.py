import asyncio
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.workers import Worker
from validators import utils
from validators.services.redis import Redis


class TaskMgr:
    def __init__(self, uid_to_capacities, dendrite, metagraph):
        # Initialize Redis client
        self.redis_client = Redis.redis_client
        self.resources = {}
        self.init_resources(uid_to_capacities)
        self.dendrite = dendrite
        self.metagraph = metagraph

    def assign_task(self, synapse: ALL_SYNAPSE_TYPE):

        # Find the worker with the most available resources (simplified logic)
        resource_key = max(self.resources,
                           key=lambda w: self.resources[w])  # Example: Assign to worker with max remaining bandwidth
        if self.resources[resource_key] <= 0:
            bt.logging.debug(f"no available resources to assign this task.")
            return None

        task_id = utils.create_hash_value((synapse.json()))
        bt.logging.trace(f"Assigning task {task_id} to {resource_key}")

        # decrease remaining capacity after sending request.
        self.resources[resource_key] -= 1
        # Push task to the selected worker's task queue
        worker = Worker(task_id=task_id, dendrite=self.dendrite, axon=self.get_axon_from_resource_key(resource_key))
        self.redis_client.rpush(f"tasks:{task_id}", synapse.json())
        asyncio.create_task(worker.pull_and_run_task())

    def get_axon_from_resource_key(self, resource_key):
        uid = resource_key.split("_")[0]
        return self.metagraph.axons[uid]

    def init_resources(self, uid_to_capacities):
        # init resources
        for uid, cap_info in uid_to_capacities.items():
            for provider, model_to_cap in cap_info.items():
                for model, cap in model_to_cap.items():
                    resource_key = f"{uid}_{provider}_{model}"
                    self.resources[resource_key] = cap