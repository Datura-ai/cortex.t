import bittensor as bt
from cortext.protocol import StreamPrompting
from typing import Tuple

from miner.services.base import BaseService
from cortext import ISALIVE_BLACKLIST_STAKE


class IsAliveService(BaseService):
    def __init__(self, metagraph, blacklist_amt=ISALIVE_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    def forward_fn(self, synapse: StreamPrompting):
        bt.logging.debug("answered to be active")
        synapse.completion = "True"
        return synapse

    def blacklist_fn(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
