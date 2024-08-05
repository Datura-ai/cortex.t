import bittensor as bt
from cortext.protocol import StreamPrompting
from typing import Tuple

from .base import BaseService
from miner.config import config


class TextService(BaseService):
    def __init__(self, metagraph, blacklist_amt=config.BLACKLIST_AMT):
        super().__init__(metagraph, blacklist_amt)

    def forward_fn(self, synapse: StreamPrompting):
        synapse.completion = "completed by miner"
        return synapse

    def blacklist_fn(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        return False, ""
