import bittensor as bt
from anthropic import AsyncAnthropic
from starlette.types import Send

from .base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting
from miner.error_handler import error_handler


class Anthropic(Provider):
    def __init__(self, synapse):
        super().__init__(synapse)
        try:
            self.anthropic_client = AsyncAnthropic(timeout=config.ASYNC_TIME_OUT, api_key=config.ANTHROPIC_API_KEY)
        except Exception as err:
            bt.logging.error(f"api_key looks incorrect or expired. {err}")
            self.anthropic_client = None

    @error_handler
    async def _prompt(self, synapse: StreamPrompting, send: Send):
        filtered_messages, system_prompt = self.generate_messages_to_claude(self.messages)

        stream_kwargs = {
            "max_tokens": synapse.max_tokens,
            "messages": filtered_messages,
            "model": synapse.model,
        }

        if system_prompt:
            stream_kwargs["system"] = system_prompt
        completion = self.anthropic_client.messages.stream(**stream_kwargs)
        try:
            async with completion as stream:
                async for text in stream.text_stream:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": text.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.info(f"Streamed text: {text}")
        except Exception as err:
            bt.logging.error(f"{err}")
        finally:
            await send({"type": "http.response.body", "body": b'', "more_body": False})

    def image_service(self, synapse):
        pass

    def embeddings_service(self, synapse):
        pass
