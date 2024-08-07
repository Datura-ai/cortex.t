import bittensor as bt
from cortext.protocol import IsAlive
from typing import Tuple

from .base import BaseService
from cortext import ISALIVE_BLACKLIST_STAKE


class IsAliveService(BaseService):
    def __init__(self, metagraph, blacklist_amt=ISALIVE_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    async def forward_fn(self, synapse: IsAlive):
        bt.logging.debug("answered to be active")
        synapse.completion = "True"
        return synapse

    async def blacklist_fn(self, synapse: IsAlive) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
