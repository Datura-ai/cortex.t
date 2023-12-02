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
import threading
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
available_uids = []
lock = threading.Lock()
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
    wandb_runs[project] = run

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


def update_weights(total_scores, steps_passed, validators, config, subtensor, wallet, metagraph):
    # Update weights based on total scores
    if steps_passed % len(validators) == len(validators) - 1:
        avg_scores = total_scores / (steps_passed + 1)
        set_weights(avg_scores, config, subtensor, wallet, metagraph)


async def check_uid(dendrite, axon, uid):
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=4)
        bt.logging.debug(f"UID {uid} {'is' if response.is_success else 'is not'} active")
        return uid if response.is_success else None
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {traceback.format_exc()}")
        return None

async def get_and_update_available_uids(dendrite, metagraph, available_uids, loop):
    asyncio.set_event_loop(loop)
    await asyncio.sleep(5)
    while True:
        try:
            bt.logging.info("Checking available_uids")
            tasks = [check_uid(dendrite, metagraph.axons[uid.item()], uid.item()) for uid in metagraph.uids]
            updated_uids = [uid for uid in await asyncio.gather(*tasks) if uid is not None]
            available_uids[:] = updated_uids
            bt.logging.info(f"Available UIDs: {available_uids}")
            bt.logging.info(f"Sleeping at: {time.strftime('%X')}")
            await asyncio.sleep(60)
            bt.logging.info(f"Woke up at: {time.strftime('%X')}")
        except Exception as e:
            bt.logging.error(f"UID update exception: {traceback.format_exc()}")


def run_async_in_thread(func, *args):
    def thread_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        args_with_loop = (*args, loop)
        try:
            loop.run_until_complete(func(*args_with_loop))
        finally:
            loop.close()
    return thread_task

async def validator_thread(config, dendrite, metagraph, validator, total_scores, steps_passed, available_uids, lock, loop):
    asyncio.set_event_loop(loop)
    bt.logging.info(f"Starting validator_thread for {validator.__class__.__name__}")
    while True:
        bt.logging.info(f"Available UIDs for  {validator.__class__.__name__}: {available_uids}")
        if available_uids:
            lock.acquire()
            try:
                scores, uid_scores_dict, wandb_data = await process_modality(config, dendrite, metagraph, validator, available_uids)
                total_scores += scores
                steps_passed[validator] += 1
                await asyncio.sleep(100)
            except Exception as e:
                bt.logging.error(f"Thread exception: {traceback.format_exc()}")
                break
            finally:
                lock.release()
        else:
            await asyncio.sleep(5)
            
async def process_modality(config, dendrite, metagraph, validator, available_uids):
    bt.logging.info(f"Processing for {validator.__class__.__name__} with UIDs: {available_uids}")
    return await validator.get_and_score(available_uids)

async def query_synapse(dendrite, metagraph, subtensor, config, wallet, available_uids):
    lock = threading.Lock()
    validators = [text_vali, image_vali, embed_vali][:2]
    steps_passed = {validator: 0 for validator in validators}
    total_scores = torch.zeros(len(metagraph.hotkeys))

    for validator in validators:
        thread_task = run_async_in_thread(validator_thread, config, dendrite, metagraph, validator, total_scores, steps_passed, available_uids, lock)
        threading.Thread(target=thread_task).start()

    while True:
        try:
            metagraph = await sync_metagraph(subtensor, config)
            lock.acquire() 
            try:
                total_steps = sum(steps_passed.values())
                if total_steps % len(validators) == len(validators) - 1:
                    avg_scores = total_scores / total_steps
                    set_weights(avg_scores, config, subtensor, wallet, metagraph)
            finally:
                lock.release()
        except Exception as e:
            bt.logging.error(f"General exception: {traceback.format_exc()}")
            await asyncio.sleep(100)

def main():
    config = get_config()
    wallet, subtensor, dendrite, metagraph, my_uid = initialize_components(config)
    available_uids = []
    initialize_validators({
        "dendrite": dendrite,
        "metagraph": metagraph,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet
    })
    init_wandb(config, my_uid, wallet)
    thread_task = run_async_in_thread(get_and_update_available_uids, dendrite, metagraph, available_uids)
    threading.Thread(target=thread_task).start()
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet, available_uids))

if __name__ == "__main__":
    main()