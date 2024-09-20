from collections import defaultdict

from typing import Union, AsyncGenerator, Any, List
from pydantic import BaseModel
import bittensor
from bittensor import dendrite, axon
import bittensor as bt
from cortext import ALL_SYNAPSE_TYPE, MIN_REQUEST_PERIOD


class Request(BaseModel):
    target_axon: Union[bittensor.AxonInfo, bittensor.axon]
    synapse: ALL_SYNAPSE_TYPE = bittensor.Synapse(),
    timeout: float = 12.0,
    deserialize: bool = True


class Dendrite(dendrite):
    # class variable to store all status of miners.
    hotkey_to_uid_capacity = defaultdict(tuple)
    requests_queue: List[Request] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def push_request_queue(cls, request):
        cls.push_request_queue(request)


    @classmethod
    def get_remaining_capacity(cls, target_axon: axon, synapse: ALL_SYNAPSE_TYPE):
        hotkey = target_axon.info().hotkey
        uid, cap = cls.miners_to_capacity[hotkey]
        provider = synapse.provider
        model = synapse.model
        return uid, cap.get(provider).get(model)

    @classmethod
    def decrease_capacity(cls, target_axon: axon, synapse: ALL_SYNAPSE_TYPE):
        pass

    async def call_stream(
            self,
            target_axon: Union[bittensor.AxonInfo, bittensor.axon],
            synapse: ALL_SYNAPSE_TYPE = bittensor.Synapse(),  # type: ignore
            timeout: float = 12.0,
            deserialize: bool = True
    ) -> AsyncGenerator[Any, Any]:
        uid, remain_cap = Dendrite.get_remaining_capacity(target_axon, synapse)
        if remain_cap > 0:
            # decrease capacity by one as it's used.

            return super().call_stream(target_axon, synapse, timeout, deserialize)
        else:
            bt.logging.debug(f"remain_cap is {remain_cap} for this uid {uid}. so can't send request.")
            raise StopAsyncIteration

    async def call(
            self,
            target_axon: Union[bittensor.AxonInfo, bittensor.axon],
            synapse: ALL_SYNAPSE_TYPE = bittensor.Synapse(),
            timeout: float = 12.0,
            deserialize: bool = True,
    ) -> bittensor.Synapse:
        uid, remain_cap = Dendrite.get_remaining_capacity(target_axon, synapse)
        if remain_cap > 0:
            return await super().call(target_axon, synapse, timeout, deserialize)
        else:
            bt.logging.debug(f"remain_cap is {remain_cap} for this uid {uid}. so can't send request.")
            return synapse
