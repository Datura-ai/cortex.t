import bittensor as bt
import argparse
import traceback
import template
import asyncio
from template.protocol import StreamPrompting

def initialize():
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(18)
    return config, wallet, subtensor, dendrite, metagraph

async def query_synapse(dendrite, metagraph, subtensor, prompt):
    try:
        metagraph = subtensor.metagraph( 18 )
        axon = metagraph.axons[9]
        syn = StreamPrompting(roles=["user"], messages=[prompt], engine = "gpt-3.5-turbo")

        async def main():
            full_response = ""
            responses = await dendrite([axon], syn, deserialize=False, streaming=True)
            for resp in responses:
                i = 0
                async for chunk in resp:
                    i += 1
                    if isinstance(chunk, list):
                        print(chunk[0], end="", flush=True)
                        full_response += chunk[0]
                    else:
                        synapse = chunk
                break
            print("\n")
            return full_response
        full_response = await main()

    except Exception as e:
        bt.logging.info(f"General exception at step: {e}\n{traceback.format_exc()}")

def main():
    config, wallet, subtensor, dendrite, metagraph = initialize()
    prompt = "tell me a long story about bad coworkers"
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, prompt))

if __name__ == "__main__":
    main()
