import bittensor as bt
import asyncio
import traceback

from cortext.protocol import StreamPrompting, Bandwidth, IsAlive



async def query_miner(dendrite: bt.dendrite, axon_to_use, synapse, timeout, streaming):
    try:
        # print(f"calling vali axon {axon_to_use} to miner uid {synapse.uid} for query {synapse.messages}")
        if streaming is False:
            responses = await dendrite.call(
                target_axon=axon_to_use,
                synapse=synapse,
                timeout=timeout,
                deserialize=False
            )
            return responses
        else:
            responses = dendrite.call_stream(
                target_axon=axon_to_use,
                synapse=synapse,
                timeout=timeout,
            )
            return await handle_response(responses)
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        return None


async def handle_response(response):
    full_response = ""
    try:
        async for chunk in response:
            if isinstance(chunk, str):
                full_response += chunk
            else:
                # print(f"\n\nFinal synapse: {chunk}\n")
                pass
    except Exception as e:
        print(f"Error processing response for uid {e}")
    return full_response


async def main():
    print("synching metagraph, this takes way too long.........")
    subtensor = bt.subtensor(network="test")
    meta = subtensor.metagraph(netuid=196)
    print("metagraph synched!")

    # This needs to be your validator wallet that is running your subnet 18 validator
    wallet = bt.wallet(name="miner", hotkey="default")
    dendrite = bt.dendrite(wallet=wallet)
    vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    axon_to_use = meta.axons[vali_uid]
    print(f"axon to use: {axon_to_use}")

    # This is the question to send your validator to send your miner.
    prompt = "Give me a story about a cat"
    messages = [{'role': 'user', 'content': prompt}]

    # see options for providers/models here: https://github.com/Datura-ai/cortex.t/blob/34f0160213d26a829e9619e3df9441760a0da1ad/cortext/constants.py#L10
    synapse = StreamPrompting(
        messages=messages,
        provider="OpenAI",
        model="gpt-4o",
    )
    timeout = 60
    streaming = True
    # synapse = Bandwidth()
    # timeout = 60
    # streaming = False
    # print("querying miner")
    tasks = []
    import time
    start_time = time.time()

    from copy import deepcopy
    # for i in range(1):
    #     tasks.append(query_miner(dendrite, axon_to_use, synapse, timeout, streaming))
    # print(time.time() - start_time)

    results = await query_miner(dendrite, axon_to_use, synapse, timeout, streaming)
    print(time.time() - start_time)
    print(results)
    # print("acerr", results[0])


if __name__ == "__main__":
    asyncio.run(main())
