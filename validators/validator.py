import os
import time
import torch
import wandb
import string
import random
import asyncio
import template
import argparse
import traceback
import bittensor as bt
import template.utils as utils

from fastapi import FastAPI
from template.protocol import IsAlive
from base_validator import BaseValidator
from text_validator import TextValidator
from image_validator import ImageValidator
from embeddings_validator import EmbeddingsValidator


moving_average_scores = None
text_vali = None
image_vali = None
embed_vali = None
app = FastAPI()


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


def init_wandb(config, my_uid, wallet):
    if config.wandb_on:
        run_name = f'validator-{my_uid}-{template.__version__}'
        config.uid = my_uid
        config.hotkey = wallet.hotkey.ss58_address
        config.run_name = run_name
        config.version = template.__version__
        config.type = 'validator'

        for project in template.PROJECT_NAMES:
            init_specific_wandb(project, my_uid, config, run_name, wallet)

        bt.logging.success("Started all wandb runs")


def init_specific_wandb(project, my_subnet_uid, config, run_name, wallet):
    run = wandb.init(
        name=run_name,
        project=project,
        entity='cortex-t',
        config=config,
        dir=config.full_path,
        reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature 
    wandb.config.update(config, allow_val_change=True)

def initialize_components(config):
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    dendrite = bt.dendrite(wallet=wallet)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register --netuid 18 and try again.")
        exit()

    return wallet, subtensor, dendrite, metagraph, my_uid


def initialize_validators(vali_config):
    global text_vali, image_vali, embed_vali

    text_vali = TextValidator(**vali_config)
    image_vali = ImageValidator(**vali_config)
    # embed_vali = EmbeddingsValidator(**vali_config)


@app.post("/text-validator/")
async def text_validator_endpoint(data: str): 
    try:
        validation_result = await text_vali.get_and_score(request_data.text)
        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image-validator/")
async def image_validator_endpoint(data: str): 
    try:
        validation_result = await image_vali.get_and_score(request_data.text)
        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings-validator/")
async def embeddings_validator_endpoint(data: str):
    try:
        validation_result = await embed_vali.get_and_score(request_data.text)
        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    
async def sync_metagraph(subtensor, config):
    return subtensor.metagraph(config.netuid)


async def process_modality(dendrite, metagraph, validators, available_uids, steps_passed):
    # Calculate and return scores and uid_scores_dict
    validator_index = steps_passed % len(validators)
    validator = validators[validator_index]
    scores, uid_scores_dict = await validator.get_and_score(available_uids)
    return scores, uid_scores_dict


def update_weights(total_scores, steps_passed, validators, config, subtensor, wallet, metagraph):
    # Update weights based on total scores
    if steps_passed % len(validators) == len(validators) - 1:
        avg_scores = total_scores / (steps_passed + 1)
        set_weights(avg_scores, config, subtensor, wallet, metagraph)


async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    steps_passed = 0
    total_scores = torch.zeros(len(metagraph.hotkeys))
    # validators to use in the loop
    validators = [text_vali, image_vali, embed_vali]
    validators = validators[:2]

    while True:
        try:
            metagraph = await sync_metagraph(subtensor, config)
            available_uids = await get_available_uids(dendrite, metagraph)
            bt.logging.info(f"available_uids = {available_uids}")

            if not available_uids:
                time.sleep(5)
                continue

            scores, uid_scores_dict = await process_modality(dendrite, metagraph, validators, available_uids, steps_passed)
            total_scores += scores
            update_weights(total_scores, steps_passed, validators, config, subtensor, wallet, metagraph)

            steps_passed += 1
            time.sleep(3)

        except Exception as e:
            bt.logging.info(f"General exception: {e}\n{traceback.format_exc()}")
            time.sleep(100)


def main():
    global validators
    config = get_config()
    wallet, subtensor, dendrite, metagraph, my_uid = initialize_components(config)
    validator_config = {
        "dendrite": dendrite,
        "metagraph": metagraph,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet
    }
    initialize_validators(validator_config)
    init_wandb(config, my_uid, wallet)
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(query_synapse(dendrite, metagraph, subtensor, config, wallet))

    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        state = utils.get_state()
        utils.save_state_to_file(state)
        if config.wandb_on: wandb.finish()

    finally:
        loop.close()


if __name__ == "__main__":
    main()