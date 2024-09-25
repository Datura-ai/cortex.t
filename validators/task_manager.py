import asyncio
from copy import deepcopy
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.utils import error_handler
from validators.workers import Worker
from validators import utils


class TaskMgr:
    def __init__(self, uid_to_capacities, dendrite, metagraph, redis_client):
        # Initialize Redis client
        self.redis_client = redis_client
        self.remain_resources = deepcopy(uid_to_capacities)
        self.dendrite = dendrite
        self.metagraph = metagraph

    @error_handler
    def assign_task(self, synapse: ALL_SYNAPSE_TYPE):
        # find miner which bandwidth > 0.
        uid = self.choose_miner(synapse)  # Example: Assign to worker with max remaining bandwidth
        if uid is None:
            bt.logging.debug(f"no available resources to process this request.")
            return None

        synapse.uid = uid
        task_id = utils.create_hash_value((synapse.json()))
        synapse.task_id = task_id

        bt.logging.trace(f"Assigning task {task_id} to miner {uid}")

        # Push task to the selected worker's task queue
        worker = Worker(synapse=synapse, dendrite=self.dendrite, axon=self.get_axon_from_uid(uid=uid),
                        redis_client=self.redis_client)
        asyncio.create_task(worker.run_task())
        return task_id

    def get_axon_from_uid(self, uid):
        uid = int(uid)
        return self.metagraph.axons[uid]

    def choose_miner(self, synapse: ALL_SYNAPSE_TYPE):
        provider = synapse.provider
        model = synapse.model
        for uid, capacity in self.remain_resources.items():
            bandwidth = capacity.get(provider).get(model)
            if bandwidth is not None and bandwidth > 0:
                # decrease resource by one after choosing this miner for the request.
                capacity[provider][model] -= 1
                return uid
