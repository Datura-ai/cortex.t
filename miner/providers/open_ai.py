import bittensor as bt
from openai import AsyncOpenAI
from starlette.types import Send

from miner.providers.base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting


class OpenAI(Provider):
    def __init__(self):
        super().__init__()
        self.openai_client = AsyncOpenAI(timeout=config.ASYNC_TIME_OUT, api_key=config.OPENAI_API_KEY)

    async def _prompt(self, synapse: StreamPrompting, send: Send):
        provider = synapse.provider
        model = synapse.model
        messages = synapse.messages
        seed = synapse.seed
        temperature = synapse.temperature
        max_tokens = synapse.max_tokens
        top_p = synapse.top_p
        top_k = synapse.top_k

        message = messages[0]
        filtered_messages = [
            {
                "role": message["role"],
                "content": [],
            }
        ]
        if message.get("content"):
            filtered_messages[0]["content"].append(
                {
                    "type": "text",
                    "text": message["content"],
                }
            )
        if message.get("image"):
            image_url = message.get("image")
            filtered_messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            )
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=filtered_messages,
            temperature=temperature,
            stream=True,
            seed=seed,
            max_tokens=max_tokens
        )
        buffer = []
        n = 1
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            buffer.append(token)
            if len(buffer) == n:
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": True,
                    }
                )
                bt.logging.info(f"Streamed tokens: {joined_buffer}")
                buffer = []

        if buffer:
            joined_buffer = "".join(buffer)
            await send(
                {
                    "type": "http.response.body",
                    "body": joined_buffer.encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(f"Streamed tokens: {joined_buffer}")

    def image_service(self):
        pass

    def embeddings_service(self):
        pass
