import bittensor as bt
from anthropic_bedrock import AsyncAnthropicBedrock
from starlette.types import Send

from .base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting
from miner.error_handler import error_handler

class AnthropicBedrock(Provider):
    def __init__(self, synapse):
        super().__init__(synapse)
        bedrock_client_parameters = {
            "service_name": 'bedrock-runtime',
            "aws_access_key_id": config.AWS_ACCESS_KEY,
            "aws_secret_access_key": config.AWS_SECRET_KEY,
            "region_name": "us-east-1"
        }

        self.anthropic_bedrock_client = AsyncAnthropicBedrock(timeout=config.ASYNC_TIME_OUT,
                                                              **bedrock_client_parameters)

    @error_handler
    async def _prompt(self, synapse: StreamPrompting, send: Send):
        stream = await self.anthropic_bedrock_client.completions.create(
            prompt=f"\n\nHuman: {self.messages}\n\nAssistant:",
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature,  # must be <= 1.0
            top_k=self.top_k,
            top_p=self.top_p,
            model=self.model,
            stream=True,
        )

        async for completion in stream:
            if completion.completion:
                await send(
                    {
                        "type": "http.response.body",
                        "body": completion.completion.encode("utf-8"),
                        "more_body": True,
                    }
                )
                bt.logging.info(f"Streamed text: {completion.completion}")

        # Send final message to close the stream
        await send({"type": "http.response.body", "body": b'', "more_body": False})

    def image_service(self, synapse):
        pass

    def embeddings_service(self, synapse):
        pass
