import bittensor as bt
import argparse
import traceback
import template
import asyncio
from template.protocol import StreamPrompting, ImageResponse

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

# axons = [67, 36, 80, 255]

async def query_synapse_image(dendrite, metagraph, subtensor):
    try:
        axon = metagraph.axons[67]
        engine = "dall-e-3"
        size = "1024x1024"
        quality = "standard"
        style = "vivid"
        messages = "a beautiful garden outside a rural house juxtaposed on a rainy day"

        syn = ImageResponse(messages=messages, engine=engine, size=size, quality=quality, style=style)

        async def main():
            responses = await dendrite([axon], syn, deserialize=False, timeout=50)
            full_response = []  # Initialize full_response
            for resp in responses:
                print(resp)
                full_response.append(resp)
                
            return full_response

        full_response = await main()
    except Exception as e:
        bt.logging.error(f"General exception at step: {e}\n{traceback.format_exc()}")


async def query_synapse_text(dendrite, metagraph, subtensor):
    try:
        axon = metagraph.axons[67]
        prompt = "tell me a long story about bad coworkers"
        syn = StreamPrompting(messages=[{"role": "user", "content": prompt}], engine="gpt-4-1106-preview", seed=1234)

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
        bt.logging.error(f"General exception at step: {e}\n{traceback.format_exc()}")

def main():
    config, wallet, subtensor, dendrite, metagraph = initialize()
    # asyncio.run(query_synapse_text(dendrite, metagraph, subtensor))
    asyncio.run(query_synapse_image(dendrite, metagraph, subtensor))

if __name__ == "__main__":
    main()