import bittensor as bt
import google.generativeai as genai
from starlette.types import Send

from .base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting


class Gemini(Provider):
    def __init__(self, synapse):
        super().__init__(synapse)
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.genai = genai
        self.genai_model = self.genai.GenerativeModel(self.model)

    async def _prompt(self, synapse: StreamPrompting, send: Send):
        stream = []
        try:
            stream = self.genai_model.generate_content(
                str(self.messages),
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            )
        except Exception as err:
            bt.logging.exception(err)

        for chunk in stream:
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk.text.encode("utf-8"),
                    "more_body": True,
                }
            )
            bt.logging.info(f"Streamed text: {chunk.text}")

        # Send final message to close the stream
        await send({"type": "http.response.body", "body": b'', "more_body": False})

    def image_service(self, synapse):
        pass

    def embeddings_service(self, synapse):
        pass
