import asyncio
import bittensor as bt
from openai import AsyncOpenAI
from starlette.types import Send
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from .base import Provider
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
        await send(
            {
                "type": "http.response.body",
                "body": b'',
                "more_body": False,
            }
        )

    async def image_service(self, synapse):
        image_data = {}
        meta = await self.openai_client.images.generate(
            model=self.model,
            prompt=self.messages,
            size=self.size,
            quality=self.quality,
            style=self.style
        )
        image_url = meta.data[0].url
        image_revised_prompt = meta.data[0].revised_prompt
        image_data["url"] = image_url
        image_data["image_revised_prompt"] = image_revised_prompt
        bt.logging.info(f"returning image response of {image_url}")
        synapse.completion = image_data
        return synapse

    async def embeddings_service(self, synapse):
        batched_embeddings = await self.get_embeddings_in_batch(self.texts, self.model)
        synapse.embeddings = batched_embeddings
        # synapse.embeddings = [np.array(embed) for embed in batched_embeddings]
        bt.logging.info(f"synapse response is {synapse.embeddings[0][:10]}")
        return synapse

    async def get_embeddings_in_batch(self, texts, model, batch_size=10):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        tasks = []
        for batch in batches:
            filtered_batch = [text for text in batch if text.strip()]
            if filtered_batch:
                task = asyncio.create_task(self.openai_client.embeddings.create(
                    input=filtered_batch, model=model, encoding_format='float'
                ))
                tasks.append(task)
            else:
                bt.logging.info("Skipped an empty batch.")

        all_embeddings = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                bt.logging.error(f"Error in processing batch: {result}")
            else:
                batch_embeddings = [item.embedding for item in result.data]
                all_embeddings.extend(batch_embeddings)
        return all_embeddings
