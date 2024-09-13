import bittensor as bt

from cortext.protocol import Bandwidth
from typing import Tuple

from .base import BaseService
from cortext import ISALIVE_BLACKLIST_STAKE
from miner.constants import capacity_to_task_and_provider


class CapacityService(BaseService):
    def __init__(self, metagraph, blacklist_amt=ISALIVE_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    async def forward_fn(self, synapse: Bandwidth):
        bt.logging.debug("capacity request is being processed")
        synapse.bandwidth_rpm = capacity_to_task_and_provider
        bt.logging.info("check status is executed.")
        return synapse

    def blacklist_fn(self, synapse: Bandwidth) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
