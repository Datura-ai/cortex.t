import bittensor as bt
from cortext.protocol import StreamPrompting
from typing import Tuple

from miner.services.base import BaseService


class Prompt(BaseService):
    def __init__(self, metagraph, request_timestamps, blacklist_amt):
        super().__init__(metagraph, request_timestamps, blacklist_amt)

    def forward_fn(self, synapse: StreamPrompting):
        provider = synapse.provider



    def blacklist_fn(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
