from abc import ABC
import base64
from functools import partial
import httpx
from starlette.types import Send
from abc import abstractmethod

import bittensor as bt


class Provider:
    def __init__(self):
        pass

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
