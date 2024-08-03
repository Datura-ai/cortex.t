import bittensor as bt
from openai import AsyncOpenAI
from starlette.types import Send
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from miner.providers.base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting


class OpenAI(Provider):
    def __init__(self, synapse):
        super().__init__(synapse)
        self.openai_client = AsyncOpenAI(timeout=config.ASYNC_TIME_OUT, api_key=config.OPENAI_API_KEY)

    async def _prompt(self, synapse: StreamPrompting, send: Send):

        message = self.messages[0]

        filtered_message: ChatCompletionMessageParam = {
            "role": message["role"],
            "content": [],
        }

        if message.get("content"):
            filtered_message["content"].append(
                {
                    "type": "text",
                    "text": message["content"],
                }
            )
        if message.get("image"):
            image_url = message.get("image")
            filtered_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            )
        response = await self.openai_client.chat.completions.create(
            model=self.model, messages=[filtered_message],
            temperature=self.temperature, stream=True,
            seed=self.seed,
            max_tokens=self.max_tokens,
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
