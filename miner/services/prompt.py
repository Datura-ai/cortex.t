import bittensor as bt
from cortext.protocol import StreamPrompting
from cortext import PROMPT_BLACKLIST_STAKE
from typing import Tuple

from .base import BaseService


class PromptService(BaseService):
    def __init__(self, metagraph, blacklist_amt=PROMPT_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    def forward_fn(self, synapse: StreamPrompting):
        provider = self.get_instance_of_provider(synapse.provider)
        service = provider.prompt_service if provider is not None else None
        return service

    def blacklist_fn(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
