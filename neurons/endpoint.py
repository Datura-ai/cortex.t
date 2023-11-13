import bittensor as bt
import os
import torch
import argparse
import traceback
import template
import asyncio
from template.protocol import StreamPrompting, IsAlive

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument( '--wandb.on', action='store_true', help='Turn on wandb logging.')
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

def initialize_components(config):
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(config.netuid)
    return wallet, subtensor, dendrite, metagraph

async def query_synapse(dendrite, metagraph):
    metagraph.sync()
    available_uids = [uid.item() for uid in metagraph.uids]
    axon = metagraph.axons[available_uids[0]]  # Select the first available axon

    query = "What is Bittensor?"
    syn = StreamPrompting(roles=["user"], messages=[query], engine="gpt-3.5-turbo")

    async with dendrite([axon], syn, deserialize=False, streaming=True) as responses:
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, list):
                    print(chunk[0], end="", flush=True)
                else:
                    print("\n")
                break

def main(config):
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    asyncio.run(query_synapse(dendrite, metagraph))

if __name__ == "__main__":
    main(get_config())
