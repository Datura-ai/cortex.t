import bittensor as bt
from cortext.protocol import TextPrompting
from typing import Tuple

from .base import BaseService
from miner.config import config


class TextService(BaseService):
    def __init__(self, metagraph, blacklist_amt=config.BLACKLIST_AMT):
        super().__init__(metagraph, blacklist_amt)

    async def forward_fn(self, synapse: TextPrompting):
        synapse.completion = "completed by miner"
        bt.logging.info("text service is executed.")
        return synapse

    def blacklist_fn(self, synapse: TextPrompting) -> Tuple[bool, str]:
        return False, ""
