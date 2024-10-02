import asyncio
import random
from copy import deepcopy
import bittensor as bt

from cortext import ALL_SYNAPSE_TYPE
from validators.utils import error_handler
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
        self.remain_resources = deepcopy(self.uid_to_capacity)
        bt.logging.debug(f"resource is restored. remain_resources = {self.remain_resources}")

    def get_remaining_bandwidth(self, uid, provider, model):
        if self.remain_resources.get(uid):
            if self.remain_resources.get(uid).get(provider):
                return self.remain_resources.get(uid).get(provider).get(model)


    def update_remain_capacity_based_on_new_capacity(self, new_uid_to_capacity):
        for uid, capacity in new_uid_to_capacity.items():
            if not capacity:
                continue
            for provider, model_to_cap in capacity.items():
                for model, cap in model_to_cap.items():
                    if self.get_remaining_bandwidth(uid, provider, model) is None:
                        utils.update_nested_dict(self.remain_resources, keys=[uid, provider, model], value=cap)
                    else:
                        diff = self.uid_to_capacity[uid][provider][model] - cap
                        if diff:
                            bt.logging.debug(f"diff {diff} found in {uid}, {provider}, {model}")
                        self.remain_resources[uid][provider][model] -= diff
        bt.logging.debug(f"remain_resources after epoch = {self.remain_resources}")

    @error_handler
    def assign_task(self, synapse: ALL_SYNAPSE_TYPE):
        # find miner which bandwidth > 0.
        uid = self.choose_miner(synapse)  # Example: Assign to worker with max remaining bandwidth
        if uid is None:
            bt.logging.debug(f"no available resources to process this request.")
            return None
        bt.logging.trace(f"Assigning task to miner {uid}")
        return uid

    def get_axon_from_uid(self, uid):
        uid = int(uid)
        return self.metagraph.axons[uid]

    def choose_miner(self, synapse: ALL_SYNAPSE_TYPE):
        provider = synapse.provider
        model = synapse.model
        available_uids = []
        for uid in self.remain_resources:
            capacity = self.remain_resources.get(uid)
            bandwidth = capacity.get(provider).get(model)
            if bandwidth is not None and bandwidth > 0:
                # decrease resource by one after choosing this miner for the request.
                available_uids.append(uid)
        uid = random.choice(available_uids)
        self.remain_resources[uid][provider][model] -= 1
        return uid
