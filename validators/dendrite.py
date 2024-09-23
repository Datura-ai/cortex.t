import asyncio
from collections import defaultdict

from typing import Union, AsyncGenerator, Any, List
from enum import Enum

from pydantic import BaseModel
import bittensor
from bittensor import dendrite, axon
import bittensor as bt
from cortext import ALL_SYNAPSE_TYPE, MIN_REQUEST_PERIOD


class RequestType(str, Enum):  # Inherit from str to enforce the value type as string
    organic_type = 'organic'
    synthetic_type = 'synthetic'


class Request(BaseModel):
    target_axon: Union[bittensor.AxonInfo, bittensor.axon]
    synapse: ALL_SYNAPSE_TYPE = bittensor.Synapse()
    timeout: float = 12.0
    deserialize: bool = True
    type: RequestType
    stream: False
    request_id: int


class Dendrite(dendrite):
    # class variable to store all status of miners.
    hotkey_to_uid_capacity = defaultdict(tuple)
    synthetic_requests_queue: List[Request] = []
    organic_requests_queue: List[Request] = []


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def push_request_queue(cls, request: Request):
        if request.type == RequestType.organic_type:
            cls.organic_requests_queue.append(request)
        if request.type == RequestType.synthetic_type:
            cls.synthetic_requests_queue.append(request)

    @classmethod
    async def process_requests(cls):
        while True:
            task_to_request_id = {}
            if cls.organic_requests_queue:
                # distribute organic queries to miners according to bandwidth.
                bt.logging.info("distribute organic queries to miners according to bandwidth.")
                for request in cls.organic_requests_queue:
                    task = asyncio.create_task(cls.create_task_from_request(request))
                    task_to_request_id[task] = request.request_id
            if cls.synthetic_requests_queue:
                bt.logging.info("start synthetic query and test bandwidth for all miners.")
                for request in cls.synthetic_requests_queue:
                    task = asyncio.create_task(cls.create_task_from_request(request))
                    task_to_request_id[task] = request.request_id
                pass

            for completed_task in asyncio.as_completed(task_to_request_id):
                result = await completed_task
                request_id = task_to_request_id[completed_task]
                bt.logging.info(f"request {request_id} is complete. {result}")
                # push result to redis.

            # process result
            await asyncio.sleep(1)

    @classmethod
    async def create_task_from_request(cls, request):
        uid, cap = cls.get_remaining_capacity(request)
        if cap > 0:
            if request.stream:
                task = super().call_stream(target_axon=request.target_axon, synapse=request.synapse,
                                           timeout=request.timeout,
                                           deserialize=request.deserialize)
            else:
                task = await super().call(target_axon=request.target_axon, synapse=request.synapse,
                                    timeout=request.timeout,
                                    deserialize=request.deserialize)
            cls.decrease_remaining_cap_after_request(request)
            return task
        bt.logging.info(f"can't process this request because all miners are too busy now")
        return None

    @classmethod
    def decrease_remaining_cap_after_request(cls, request):
        target_axon = request.target_axon
        synapse = request.synapse
        hotkey = target_axon.info().hotkey
        uid, cap = cls.hotkey_to_uid_capacity[hotkey]
        provider = synapse.provider
        model = synapse.model
        cap[provider][model] = cap - 1

        # update capacity with decreased by 1 for provider, model
        cls.hotkey_to_uid_capacity[hotkey] = uid, cap


    @classmethod
    def get_remaining_capacity(cls, request):
        target_axon = request.target_axon
        synapse = request.synapse
        hotkey = target_axon.info().hotkey
        uid, cap = cls.hotkey_to_uid_capacity[hotkey]
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
