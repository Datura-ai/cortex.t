import bittensor as bt
import google.generativeai as genai
from starlette.types import Send

from miner.providers.base import Provider
from miner.config import config
from cortext.protocol import StreamPrompting


class Gemini(Provider):
    def __init__(self):
        super().__init__()
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.genai = genai

    async def _prompt(self, synapse: StreamPrompting, send: Send):
        model = self.genai.GenerativeModel(synapse.model)
        stream = model.generate_content(
            str(synapse.messages),
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=synapse.temperature,
                top_p=synapse.top_p,
                top_k=synapse.top_k,
            )
        )
        for chunk in stream:
            for part in chunk.candidates[0].content.parts:
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

    def image_service(self):
        pass

    def embeddings_service(self):
        pass
