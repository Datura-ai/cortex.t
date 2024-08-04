import bittensor as bt
from cortext.protocol import IsAlive, TextPrompting


async def is_alive(self, synapse: IsAlive) -> IsAlive:
    bt.logging.debug("answered to be active")
    synapse.completion = "True"
    return synapse


async def text(self, synapse: TextPrompting) -> TextPrompting:
    synapse.completion = "completed by miner"
    return synapse
