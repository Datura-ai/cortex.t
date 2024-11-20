import bittensor as bt
from app.models import ChatRequest
from app.core.protocol import StreamPrompting

subtensor = bt.subtensor(network="finney")
meta = subtensor.metagraph(netuid=18)
print("metagraph synched!")

# This needs to be your validator wallet that is running your subnet 18 validator
wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
dendrite = CortexDendrite(wallet=wallet)
vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
axon_to_use = meta.axons[vali_uid]


async def handle_response(resp):
    full_response = ""
    try:
        async for chunk in resp:
            if isinstance(chunk, str):
                yield chunk
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
    except Exception as e:
        print(f"Error processing response for uid {e}")


async def query_miner(chat_request: ChatRequest):
    try:
        print(f"calling vali axon {axon_to_use} to miner uid {synapse.uid} for query {synapse.messages}")
        synapse = StreamPrompting()
        synapse.messages = chat_request.messages
        synapse.model = chat_request.model
        synapse.provider = chat_request.provider
        synapse.temperature = chat_request.temperature
        synapse.max_tokens = chat_request.max_tokens
        synapse.top_p = chat_request.top_p
        synapse.streaming = chat_request.stream

        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=60
        )
        async for chunk in resp:
            if isinstance(chunk, str):
                yield chunk
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        yield ""
