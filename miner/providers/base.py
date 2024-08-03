import base64
import bittensor as bt
from functools import partial
import httpx
from starlette.types import Send
from abc import abstractmethod

from cortext.protocol import StreamPrompting, TextPrompting, Embeddings, ImageResponse, IsAlive
from typing import Union


class Provider:
    def __init__(self, synapse: Union[StreamPrompting, TextPrompting, Embeddings, ImageResponse, IsAlive]):
        self.model = synapse.model
        self.uid = synapse.uid
        self.timeout = synapse.timeout
        if type(synapse) in [StreamPrompting, TextPrompting]:
            self.messages = synapse.messages
            self.required_hash_fields = synapse.required_hash_fields
            self.seed = synapse.seed
            self.temperature = synapse.temperature
            self.max_tokens = synapse.max_tokens
            self.top_p = synapse.top_p
            self.top_k = synapse.top_k
            self.completion = synapse.completion
            self.provider = synapse.provider
            self.streaming = synapse.streaming
        elif type(synapse) is Embeddings:
            self.texts = synapse.texts
            self.embeddings = synapse.embeddings
        elif type(synapse) is ImageResponse:
            self.completion = synapse.completion
            self.messages = synapse.messages
            self.provider = synapse.provider
            self.seed = synapse.seed
            self.samples = synapse.samples
            self.cfg_scale = synapse.cfg_scale
            self.sampler = synapse.sampler
            self.steps = synapse.steps
            self.style = synapse.style
            self.size = synapse.size
            self.height = synapse.height
            self.width = synapse.width
            self.quality = synapse.quality
            self.required_hash_fields = synapse.required_hash_fields
        elif type(synapse) is IsAlive:
            self.answer = synapse.answer
            self.completion = synapse.completion
        else:
            bt.logging.error(f"unknown synapse {type(synapse)}")

    @staticmethod
    def generate_messages_to_claude(messages: list):
        system_prompt = None
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                message_to_append = {
                    "role": message["role"],
                    "content": [],
                }
                if message.get("image"):
                    image_url = message.get("image")
                    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
                    message_to_append["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        }
                    )
                if message.get("content"):
                    message_to_append["content"].append(
                        {
                            "type": "text",
                            "text": message["content"],
                        }
                    )
                filtered_messages.append(message_to_append)
        return filtered_messages, system_prompt

    async def chat_service(self, synapse: bt.StreamingSynapse):
        token_streamer = partial(self._prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    @abstractmethod
    async def _prompt(self, synapse, send: Send):
        pass

    @abstractmethod
    def image_service(self, synapse: bt.Synapse):
        pass

    @abstractmethod
    def embeddings_service(self, synapse: bt.Synapse):
        pass
