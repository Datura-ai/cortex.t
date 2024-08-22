import aioboto3
import json
import bittensor as bt
from starlette.types import Send

from .base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting


class Bedrock(Provider):
    def __init__(self, synapse):
        super().__init__(synapse)
        self.bedrock_client_parameters = {
            "service_name": 'bedrock-runtime',
            "aws_access_key_id": config.AWS_ACCESS_KEY,
            "aws_secret_access_key": config.AWS_SECRET_KEY,
            "region_name": "us-east-1"
        }
        self.aws_session = aioboto3.Session()

    async def generate_request(self):
        native_request = None
        if self.model.startswith("cohere"):
            native_request = {
                "message": self.messages[0]["content"],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "p": self.top_p,
                "seed": self.seed,
            }
        elif self.model.startswith("meta"):
            native_request = {
                "prompt": self.messages[0]["content"],
                "temperature": self.temperature,
                "max_gen_len": 2048 if self.max_tokens > 2048 else self.max_tokens,
                "top_p": self.top_p,
            }
        elif self.model.startswith("anthropic"):
            message_, system_prompt = await self.generate_messages_to_claude(self.messages)
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": message_,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
            if system_prompt:
                native_request["system"] = system_prompt
        elif self.model.startswith("mistral"):
            native_request = {
                "prompt": self.messages[0]["content"],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        elif self.model.startswith("amazon"):
            native_request = {
                "inputText": self.messages[0]["content"],
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p,
                },
            }
        elif self.model.startswith("ai21"):
            native_request = {
                "prompt": self.messages[0]["content"],
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            }
        request_body = json.dumps(native_request)
        return request_body

    async def extract_token(self, chunk):
        token = None
        if self.model.startswith("cohere"):
            token = chunk.get("text") or ""
        elif self.model.startswith("meta"):
            token = chunk.get("generation") or ""
        elif self.model.startswith("anthropic"):
            token = ""
            if chunk['type'] == 'content_block_delta':
                if chunk['delta']['type'] == 'text_delta':
                    token = chunk['delta']['text']
        elif self.model.startswith("mistral"):
            token = chunk.get("outputs")[0]["text"] or ""
        elif self.model.startswith("amazon"):
            token = chunk.get("outputText") or ""
        elif self.model.startswith("ai21"):
            token = json.loads(chunk)["completions"][0]["data"]["text"]
        return token

    async def _prompt(self, synapse: StreamPrompting, send: Send):
        self.model = synapse.model
        self.messages = synapse.messages
        self.temperature = synapse.temperature
        self.max_tokens = synapse.max_tokens
        self.top_p = synapse.top_p
        self.seed = synapse.seed
        request = await self.generate_request()

        async with self.aws_session.client(**self.bedrock_client_parameters) as client:
            if self.model.startswith("ai21"):
                response = await client.invoke_model(
                    modelId=self.model, body=request
                )
                message = await response['body'].read()
                message = await self.extract_token(message)
                await send(
                    {
                        "type": "http.response.body",
                        "body": message.encode("utf-8"),
                        "more_body": True,
                    }
                )
                bt.logging.info(f"Streamed tokens: {message}")
            else:
                stream = await client.invoke_model_with_response_stream(
                    modelId=self.model, body=request
                )

                buffer = []
                n = 1
                async for event in stream["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    token = await self.extract_token(chunk)
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
                else:
                    await send({"type": "http.response.body", "body": b'', "more_body": False})


    def image_service(self, synapse):
        pass

    def embeddings_service(self, synapse):
        pass
