from abc import abstractmethod
import bittensor as bt
from typing import Tuple
import cortext
import time
import traceback
from collections import deque

from cortext import IsAlive
from miner.config import config
from miner.providers.base import Provider
from cortext.metaclasses import ProviderRegistryMeta, ServiceRegistryMeta


class BaseService(metaclass=ServiceRegistryMeta):
    request_timestamps: dict = {}

    def __init__(self, metagraph, blacklist_amt):
        self.metagraph = metagraph
        self.blacklist_amt = blacklist_amt if config.ENV == 'prod' else config.BLACKLIST_AMT

    def get_instance_of_provider(self, provider_name):
        provider_obj: Provider = ProviderRegistryMeta.get_class(provider_name)
        if provider_obj is None:
            bt.logging.info(f"{provider_name} is not supported currently in this network {self.metagraph.network}.")
            return None
        return provider_obj

    @abstractmethod
    async def forward_fn(self, synapse):
        pass

    @abstractmethod
    def blacklist_fn(self, synapse):
        pass

    @classmethod
    def get_axon_attach_funcs(cls, metagraph):
        service = cls(metagraph, config.BLACKLIST_AMT)
        return service.forward_fn, service.blacklist_fn

    def base_blacklist(self, synapse) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__
            if synapse_type == IsAlive.__name__:
                return False, "Don't blacklist for IsAlive checking Synapse"

            uid = None
            for _uid, _axon in enumerate(self.metagraph.axons):  # noqa: B007
                if _axon.hotkey == hotkey:
                    uid = _uid
                    break

            if uid is None and cortext.ALLOW_NON_REGISTERED is False:
                return True, f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}"

            # check the stake
            stake = self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]
            # if stake < self.blacklist_amt:
            #     return True, f"Blacklisted a low stake {synapse_type} request: {stake} < {self.blacklist_amt} from {hotkey}"

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")
