import bittensor as bt

from cortext.protocol import Bandwidth
from typing import Tuple

from .base import BaseService
from cortext import ISALIVE_BLACKLIST_STAKE
from cortext.constants import bandwidth_to_model


class CapacityService(BaseService):
    def __init__(self, metagraph, blacklist_amt=ISALIVE_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    async def forward_fn(self, synapse: Bandwidth):
        bt.logging.debug("capacity request is being processed")
        synapse.bandwidth_rpm = bandwidth_to_model
        bt.logging.info("check status is executed.")
        return synapse

    def blacklist_fn(self, synapse: Bandwidth) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
