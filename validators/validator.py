import argparse
import asyncio
import random
import traceback
from pathlib import Path
from typing import AsyncIterator

import bittensor as bt
import torch
import uvicorn
import wandb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from image_validator import ImageValidator
from text_validator import TextValidator

import template
from template import utils
from template.protocol import IsAlive
import sys

moving_average_scores = None
text_vali = None
image_vali = None
embed_vali = None
metagraph = None
wandb_runs = {}
app = FastAPI()
EXPECTED_ACCESS_KEY = "hello"


def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    _args = parser.parse_args()
    full_path = Path.expanduser(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    )
    config.full_path = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)
    return config


def init_wandb(config, my_uid, wallet: bt.wallet):
    if not config.wandb_on:
        return

    run_name = f'validator-{my_uid}-{template.__version__}'
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = template.__version__
    config.type = 'validator'

    # Initialize the wandb run for the single project
    run = wandb.init(
        name=run_name,
        project=template.PROJECT_NAME,
        entity='cortex-t',
        config=config,
        dir=config.full_path,
        reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{template.PROJECT_NAME}'")


def initialize_components(config: bt.config):
    global metagraph
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    dendrite = bt.dendrite(wallet=wallet)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Your validator: {wallet} is not registered to chain connection: "
            f"{subtensor}. Run btcli register --netuid 18 and try again."
        )
        sys.exit()

    return wallet, subtensor, dendrite, my_uid


def initialize_validators(vali_config):
    global text_vali, image_vali, embed_vali

    text_vali = TextValidator(**vali_config)
    image_vali = ImageValidator(**vali_config)
    # embed_vali = EmbeddingsValidator(**vali_config)
    bt.logging.info("initialized_validators")


async def check_uid(dendrite, axon, uid):
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=4)
        if response.is_success:
            bt.logging.trace(f"UID {uid} is active")
            return axon  # Return the axon info instead of the UID

        bt.logging.trace(f"UID {uid} is not active")
        return None

    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return None

async def get_available_uids(dendrite, metagraph):
    """Get a dictionary of available UIDs and their axons asynchronously."""
    tasks = {uid.item(): check_uid(dendrite, metagraph.axons[uid.item()], uid.item()) for uid in metagraph.uids}
    results = await asyncio.gather(*tasks.values())

    # Create a dictionary of UID to axon info for active UIDs
    available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

    return available_uids


def set_weights(scores, config, subtensor, wallet, metagraph):
    global moving_average_scores
    # alpha of .3 means that each new score replaces 30% of the weight of the previous weights
    alpha = .3
    if moving_average_scores is None:
        moving_average_scores = scores.clone()

    # Update the moving average scores
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores
    bt.logging.info(f"Updated moving average of weights: {moving_average_scores}")
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=moving_average_scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights.")


def update_weights(total_scores, steps_passed, config, subtensor, wallet, metagraph):
    """ Update weights based on total scores, using min-max normalization for display. """
    avg_scores = total_scores / (steps_passed + 1)

    # Normalize avg_scores to a range of 0 to 1
    min_score = torch.min(avg_scores)
    max_score = torch.max(avg_scores)

    if max_score - min_score != 0:
        normalized_scores = (avg_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = torch.zeros_like(avg_scores)

    bt.logging.info(f"normalized_scores = {normalized_scores}")
    # We can't set weights with normalized scores because that disrupts the weighting assigned to each validator class
    # Weights get normalized anyways in weight_utils
    set_weights(avg_scores, config, subtensor, wallet, metagraph)


async def process_modality(config, selected_validator, available_uids, metagraph):
    uid_list = list(available_uids.keys())
    random.shuffle(uid_list)
    bt.logging.info(f"starting {selected_validator.__class__.__name__} get_and_score for {uid_list}")
    scores, uid_scores_dict, wandb_data = await selected_validator.get_and_score(uid_list, metagraph)
    if config.wandb_on:
        wandb.log(wandb_data)
        bt.logging.success("wandb_log successful")
    return scores, uid_scores_dict


async def query_synapse(dendrite, subtensor, config, wallet):
    global metagraph
    steps_passed = 0
    total_scores = torch.zeros(len(metagraph.hotkeys))
    while True:
        try:
            metagraph = subtensor.metagraph(config.netuid)
            available_uids = await get_available_uids(dendrite, metagraph)

            if steps_passed % 5 in (0, 1, 2):
                selected_validator = text_vali
            else:
                selected_validator = image_vali

            scores, _uid_scores_dict = await process_modality(config, selected_validator, available_uids, metagraph)
            total_scores += scores

            iterations_per_set_weights = 12
            iterations_until_update = iterations_per_set_weights - ((steps_passed + 1) % iterations_per_set_weights)
            bt.logging.info(f"Updating weights in {iterations_until_update} iterations.")

            if iterations_until_update == 1:
                update_weights(total_scores, steps_passed, config, subtensor, wallet, metagraph)

            steps_passed += 1
            await asyncio.sleep(0.5)

        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)


@app.post("/text-validator/")
async def process_text_validator(request: Request, data: dict) -> StreamingResponse:
    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    async def response_stream() -> AsyncIterator[str]:
        try:
            messages_dict = {int(k): [{'role': 'user', 'content': v}] for k, v in data.items()}
            async for response in text_vali.organic(metagraph, messages_dict):
                _uid, content = response
                yield f"{content}"
        except Exception:
            bt.logging.error(f"error in response_stream {traceback.format_exc()}")

    return StreamingResponse(response_stream())

def run_fastapi() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main() -> None:
    config = get_config()
    wallet, subtensor, dendrite, my_uid = initialize_components(config)
    validator_config = {
        "dendrite": dendrite,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet
    }
    initialize_validators(validator_config)
    init_wandb(config, my_uid, wallet)

    try:
        asyncio.run(query_synapse(dendrite, subtensor, config, wallet))
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
    finally:
        state = utils.get_state()
        utils.save_state_to_file(state)
        if config.wandb_on:
            wandb.finish()

if __name__ == "__main__":
    main()
