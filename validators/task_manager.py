import asyncio
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.utils import error_handler
from validators.workers import Worker
from validators import utils


class TaskMgr:
    def __init__(self, uid_to_capacities, dendrite, metagraph, redis_client):
        # Initialize Redis client
        self.redis_client = redis_client
        self.resources = {}
        self.init_resources(uid_to_capacities)
        self.dendrite = dendrite
        self.metagraph = metagraph

    @error_handler
    def assign_task(self, synapse: ALL_SYNAPSE_TYPE):

        # Find the worker with the most available resources (simplified logic)
        resource_key = max(self.resources,
                           key=lambda w: self.resources[w])  # Example: Assign to worker with max remaining bandwidth
        if self.resources[resource_key] <= 0:
            bt.logging.debug(f"no available resources to assign this task.")
            return None
        task_id = utils.create_hash_value((synapse.json()))
        synapse.task_id = task_id
        synapse.uid = self.get_id_from_resource_key(resource_key)
        bt.logging.trace(f"Assigning task {task_id} to {resource_key}")

        # decrease remaining capacity after sending request.
        self.resources[resource_key] -= 1
        # Push task to the selected worker's task queue
        worker = Worker(synapse=synapse, dendrite=self.dendrite, axon=self.get_axon_from_resource_key(resource_key),
                        redis_client=self.redis_client)
        asyncio.create_task(worker.run_task())
        return task_id

    def get_axon_from_resource_key(self, resource_key):
        uid = int(resource_key.split("_")[0])
        return self.metagraph.axons[uid]

    @staticmethod
    def get_id_from_resource_key(resource_key):
        return int(resource_key.split("_")[0])

    def init_resources(self, uid_to_capacities):
        # init resources
        for uid, cap_info in uid_to_capacities.items():
            for provider, model_to_cap in cap_info.items():
                for model, cap in model_to_cap.items():
                    resource_key = f"{uid}_{provider}_{model}"
                    self.resources[resource_key] = cap
