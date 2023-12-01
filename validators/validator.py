import os
import re
import json
import math
import time
import wandb
import asyncio
import template
import argparse
import datetime
import traceback
import bittensor as bt
from base_validator import BaseValidator
from text_validator import TextValidator
from image_validator import ImageValidator
from embeddings_validator import EmbeddingsValidator
from template.protocol import IsAlive
import template.utils as utils

moving_average_scores = None

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

def init_wandb(my_subnet_uid):
    if config.wandb_on:
        run_name = f'validator-{my_subnet_uid}'
        config.run_name = run_name
        config.version = template.__version__
        global wandb_run
        wandb_run = wandb.init(
            name=run_name,
            anonymous="allow",
            reinit=False,
            project='synthetic-data-2',
            entity='cortex-t',
            config=config,
            dir=config.full_path,
        )
        bt.logging.success('Started wandb run')

def initialize_components(config):
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register --netuid 18 and try again.")
        exit()
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(config.netuid)
    return wallet, subtensor, dendrite, metagraph

async def check_uid(dendrite, axon, uid):
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=4)
        if response.is_success:
            bt.logging.debug(f"UID {uid} is active")
            return uid
        else:
            bt.logging.debug(f"UID {uid} is not active")
            return None
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return None

async def get_available_uids(dendrite, metagraph):
    """Get a list of available UIDs asynchronously."""
    tasks = [check_uid(dendrite, metagraph.axons[uid.item()], uid.item()) for uid in metagraph.uids]
    uids = await asyncio.gather(*tasks)
    # Filter out None values (inactive UIDs)
    return [uid for uid in uids if uid is not None]

def set_weights(scores, config, subtensor, wallet, metagraph):
    global moving_average_scores
    # alpha of .3 means that each new score has a 30% weight of the current weights
    alpha = .3
    if moving_average_scores is None:
        moving_average_scores = scores.clone()

    # Update the moving average scores
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores
    bt.logging.info(f"Updated moving average of weights: {moving_average_scores}")
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=moving_average_scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights based on moving average.")
    
async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    steps_passed = 0
    total_scores = torch.zeros(len(metagraph.hotkeys))
    image_validator = ImageValidator(dendrite, metagraph, config, subtensor, wallet)
    text_validator = TextValidator(dendrite, metagraph, config, subtensor, wallet)
    embeddings_validator = EmbeddingsValidator(dendrite, metagraph, config, subtensor, wallet)
    validators = [image_validator, text_validator, embeddings_validator]

    while True:
        try:
            # Sync metagraph and initialze scores
            metagraph = subtensor.metagraph(config.netuid)
            scores = torch.zeros(len(metagraph.hotkeys))
            uid_scores_dict = {}
            
            # Get the available UIDs
            available_uids = await get_available_uids(dendrite, metagraph)
            bt.logging.info(f"available_uids is {available_uids}")

            if not available_uids:
                time.sleep(5)
                continue

            validator_index = steps_passed / len(validators)
            validator = validators[validator_index]
            scores, uid_scores_dict = await validator.get_and_score(available_uids)


            total_scores += scores
            bt.logging.info(f"scores = {uid_scores_dict}, {2 - steps_passed % 3} iterations until set weights")
            bt.logging.info(f"total scores until set weights = {total_scores}")

            # Update weights after processing all batches
            if steps_passed % 5 == 4:
                avg_scores = total_scores / (steps_passed + 1)
                bt.logging.info(f"avg scores is {avg_scores}")
                steps_passed = 0
                set_weights(avg_scores, config, subtensor, wallet, metagraph)
                total_scores = torch.zeros(len(metagraph.hotkeys))

            steps_passed += 1
            time.sleep(100)

        except Exception as e:
            bt.logging.info(f"General exception: {e}\n{traceback.format_exc()}")
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Exiting validator.")
            state = utils.get_state()
            utils.save_state_to_file(state)
            if config.wandb_on: wandb.finish()
        finally: loop.close()

def main():
    config = get_config()
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    bt.logging.debug(f"got {wallet}, {subtensor}, {dendrite}, {metagraph}")
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    init_wandb(my_subnet_uid)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))
    return config

if __name__ == "__main__":
    asyncio.run(main())