import bittensor as bt
from cortext.protocol import ImageResponse
from typing import Tuple

from .base import BaseService
from cortext import IMAGE_BLACKLIST_STAKE


class ImageService(BaseService):
    def __init__(self, metagraph, blacklist_amt=IMAGE_BLACKLIST_STAKE):
        super().__init__(metagraph, blacklist_amt)

    async def forward_fn(self, synapse: ImageResponse):
        provider = self.get_instance_of_provider(synapse.provider)(synapse)
        bt.logging.info(f"selected image provider is {provider}")
        service = provider.image_service if provider is not None else None
        bt.logging.info("image service is executed.")
        try:
            resp = await service(synapse)
        except Exception as err:
            bt.logging.exception(err)
            return None
        return resp

    def blacklist_fn(self, synapse: ImageResponse) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist
