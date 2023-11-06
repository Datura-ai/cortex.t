# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import bittensor as bt
from starlette.responses import StreamingResponse as _StreamingResponse
from starlette.responses import Response
from starlette.types import Send, Receive, Scope
from typing import Callable, Awaitable, List
from pydantic import BaseModel
from abc import ABC, abstractmethod
import asyncio


# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class Dummy(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by recieving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.dummy_output

class Openai(bt.Synapse):
    openai_input: str
    openai_engine: str
    openai_output_stream: typing.List[str] = []

    def deserialize(self) -> str:
        return "".join(self.openai_output_stream)

class BTStreamingResponseModel(BaseModel):
    token_streamer: Callable[[Send], Awaitable[None]]

class AsyncQueueIterator:
    def __init__(self, queue):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if item is None:  # Using None as a sentinel value
            raise StopAsyncIteration
        return item

class StreamingSynapse(bt.Synapse, ABC):
    streaming_input: str
    streaming_engine: str

    class Config:
        extra = "allow"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_output = asyncio.Queue()

    async def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.stream_output.empty():
            return await self.stream_output.get()
        else:
            raise StopAsyncIteration
    
    @property
    def stream_output_iter(self):
        return AsyncQueueIterator(self.stream_output)

    class BTStreamingResponse(_StreamingResponse):
        def __init__(self, model: BTStreamingResponseModel, **kwargs):
            super().__init__(content=iter(()), **kwargs)
            self.token_streamer = model.token_streamer

        async def stream_response(self, send: Send):
            headers = [(b"content-type", b"text/event-stream")] + self.raw_headers
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await self.token_streamer(send)
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            await self.stream_response(send)
        
    def create_streaming_response(self, token_streamer: Callable[[Send], Awaitable[None]]) -> BTStreamingResponse:
        model_instance = BTStreamingResponseModel(token_streamer=token_streamer)
        return self.BTStreamingResponse(model_instance)

    async def process_streaming_response(self, response: Response):
        for chunk in response.streaming_output_stream:
            yield chunk

    def extract_response_json(self, response: Response) -> dict:
        return {"message": "".join(response.streaming_output_stream)}
