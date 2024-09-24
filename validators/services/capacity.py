import asyncio
from copy import deepcopy
from typing import List

from cortext.protocol import Bandwidth
from cortext import MIN_REQUEST_PERIOD
import bittensor as bt


class CapacityService:
    def __init__(self, metagraph, dendrite):
        self.metagraph = metagraph
        self.dendrite: bt.dendrite = dendrite
        self.timeout = 4
        self.uid_to_capacity = {}
        self.remain_uid_to_capacity = {}
        self.epoch_len = MIN_REQUEST_PERIOD

    async def query_capacity_to_miners(self, available_uids):
        capacity_query_tasks = []

        # Query all images concurrently
        for uid in available_uids:
            syn = Bandwidth()
            bt.logging.info(f"querying capacity to uid = {uid}")
            task = self.dendrite.call(self.metagraph.axons[uid], syn,
                                      timeout=self.timeout)
            capacity_query_tasks.append(task)

        # Query responses is (uid. syn)
        query_responses: List[Bandwidth] = await asyncio.gather(*capacity_query_tasks, return_exceptions=True)
        uid_to_capacity = {}
        for uid, resp in zip(available_uids, query_responses):
            if isinstance(resp, Exception):
                bt.logging.error(f"exception happens while querying capacity to miner {uid}, {resp}")
            else:
                uid_to_capacity[uid] = resp.bandwidth_rpm
        self.uid_to_capacity = deepcopy(uid_to_capacity)
        return uid_to_capacity

    async def refresh_capacity_per_epoch(self):
        while True:
            self.remain_uid_to_capacity = deepcopy(self.uid_to_capacity)
            await asyncio.sleep(self.epoch_len * 60)
