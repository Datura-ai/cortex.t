import json
from typing import AsyncGenerator
import bittensor as bt
from cursor.app.models import ChatRequest
from cursor.app.core.protocol import StreamPrompting
from cursor.app.core.config import config
from cortext.dendrite import CortexDendrite
import traceback

subtensor = bt.subtensor(network="finney")
meta = subtensor.metagraph(netuid=18)
print("metagraph synched!")

# This needs to be your validator wallet that is running your subnet 18 validator
wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
print(f"wallet_name is {config.wallet_name}, hot_key is {config.wallet_hotkey}")
dendrite = CortexDendrite(wallet=wallet)
vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
axon_to_use = meta.axons[vali_uid]


async def query_miner(chat_request: ChatRequest) -> AsyncGenerator[str, None]:
    try:
        synapse = StreamPrompting(**chat_request.dict())

        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=60
        )
        async for chunk in resp:
            if isinstance(chunk, str):
                obj = {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":chunk},"index":0,"finish_reason":None}]}
                yield "data: " + json.dumps(obj) + "\n\n"
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
        yield "[DONE]"
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        yield "Exception ocurred."

async def query_miner_no_stream(chat_request: ChatRequest):
    try:
        synapse = StreamPrompting(**chat_request.dict())

        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=60
        )
        full_resp = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                full_resp += chunk
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
        return full_resp

    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        return ""