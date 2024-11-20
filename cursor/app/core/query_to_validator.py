import bittensor as bt

subtensor = bt.subtensor(network="finney")
meta = subtensor.metagraph(netuid=18)
print("metagraph synched!")

# This needs to be your validator wallet that is running your subnet 18 validator
# wallet = bt.wallet(name="default", hotkey="default")
wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
dendrite = CortexDendrite(wallet=wallet)
vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
axon_to_use = meta.axons[vali_uid]


async def query_miner(dendrite: CortexDendrite, axon_to_use, synapse, timeout=60, streaming=True):
    try:
        print(f"calling vali axon {axon_to_use} to miner uid {synapse.uid} for query {synapse.messages}")
        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=timeout
        )
        return await handle_response(resp)
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        return None
