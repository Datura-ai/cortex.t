import asyncio
from copy import deepcopy
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.utils import error_handler
from validators.workers import Worker
from validators import utils


class TaskMgr:
    def __init__(self, uid_to_capacities, dendrite, metagraph, loop):
        # Initialize Redis client
        self.remain_resources = deepcopy(uid_to_capacities)
        self.uid_to_capacity = deepcopy(uid_to_capacities)
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.loop = loop

    def restore_capacities_for_all_miners(self):
        bt.logging.debug(f"resource is restored. self.remain_resources = {self.remain_resources}")
        self.remain_resources = deepcopy(self.uid_to_capacity)

    def update_remain_capacity_based_on_new_capacity(self, new_uid_to_capacity):
        for uid, capacity in new_uid_to_capacity.items():
            for provider, model_to_cap in capacity.items():
                for model, cap in model_to_cap.items():
                    if self.remain_resources.get(uid).get(provider).get(model) is None:
                        self.remain_resources[uid][provider][model] = cap
                    else:
                        diff = self.uid_to_capacity[uid][provider][model] - cap
                        if diff:
                            bt.logging.debug(f"diff {diff} found in {uid}, {provider}, {model}")
                        self.remain_resources[uid][provider][model] -= diff

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
        worker = Worker(synapse=synapse, dendrite=self.dendrite, axon=self.get_axon_from_uid(uid=uid))
        self.loop.create_task(worker.run_task())
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
