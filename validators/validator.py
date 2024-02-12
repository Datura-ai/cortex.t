import logging
import time
from typing import Tuple

import base  # noqa

import os
import argparse
import asyncio
import random
import traceback
from pathlib import Path

import bittensor as bt
import torch
import wandb
from aiohttp import web
from aiohttp.web_response import Response
from image_validator import ImageValidator
from embeddings_validator import EmbeddingsValidator
from text_validator import TextValidator, TestTextValidator
from base_validator import BaseValidator
from envparse import env

import template
from template import utils
import sys
import json
from weight_setter import WeightSetter, TestWeightSetter

text_vali = None
image_vali = None
embed_vali = None
metagraph = None
wandb_runs = {}
# organic requests are scored, the tasks are stored in this queue
# for later being consumed by `query_synapse` cycle:
organic_scoring_tasks = set()
EXPECTED_ACCESS_KEY = os.environ.get('EXPECTED_ACCESS_KEY', "hello")


def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.add_argument('--http_port', type=int, default=8000)
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    _args = parser.parse_args()
    full_path = Path(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    ).expanduser()
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


def initialize_validators(vali_config, test=False):
    global text_vali, image_vali, embed_vali

    text_vali = (TextValidator if not test else TestTextValidator)(**vali_config)
    image_vali = ImageValidator(**vali_config)
    embed_vali = EmbeddingsValidator(**vali_config)
    bt.logging.info("initialized_validators")


async def process_text_validator(request: web.Request):
    # Basic request validation
    if request.method != "POST" or request.path != '/text-validator/':
        return web.Response(status=400, text="Invalid request")

    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        return web.Response(status=401, text="Invalid access key")
    
    if len(validator_app.weight_setter.available_uids) == 0:
        return web.Response(status=404, text="No available UIDs")

    body = await request.json()
    messages = body['messages']

    response = web.StreamResponse()
    await response.prepare(request)

    key_to_response = {}
    uid_to_response = {}
    try:
        async for uid, key, content in text_vali.organic(metagraph=validator_app.weight_setter.metagraph, 
                                                         available_uids=validator_app.weight_setter.available_uids,
                                                         messages=messages):
            uid_to_response[uid] = uid_to_response.get(uid, '') + content
            key_to_response[key] = key_to_response.get(key, '') + content
            # await response.write(content.encode())
        prompts = {}
        for uid, message_dict in zip(uid_to_response.keys(), messages):
            (key, message_list), = message_dict.items()
            prompt = message_list[-1]['content']
            prompts[uid] = prompt  # Update prompts correctly for each uid


        validator_app.weight_setter.register_text_validator_organic_query(
            uid_to_response=uid_to_response,
            messages_dict=prompts
        )
        await response.write(json.dumps(key_to_response).encode())   
    except Exception as e:
        bt.logging.error(f'Encountered in {process_text_validator.__name__}:\n{traceback.format_exc()}, ERROR: {e}')
        await response.write(b'<<internal error>>')
    return response


class ValidatorApplication(web.Application):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.weight_setter: WeightSetter | None = None

# async def organic_scoring(request: web.Request):
#     try:
#         # Check access key
#         access_key = request.headers.get("access-key")
#         if access_key != EXPECTED_ACCESS_KEY:
#             raise web.Response(status_code=401, detail="Invalid access key")
#         body = await request.json()
#         messages = body['messages']
       
#         responses = await validator_app.weight_setter.perform_api_scoring_and_update_weights(messages)

#         return web.json_response(responses)
#     except Exception as e:
#         bt.logging.error(f'Organic scoring error: ${e}')
#         await web.Response(status_code=400, detail="{e}")

validator_app = ValidatorApplication()
validator_app.add_routes([web.post('/text-validator/', process_text_validator)])
# validator_app.add_routes([web.post('/scoring/', organic_scoring)])


def main(run_aio_app=True, test=False) -> None:
    config = get_config()
    wallet, subtensor, dendrite, my_uid = initialize_components(config)
    validator_config = {
        "dendrite": dendrite,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet
    }
    initialize_validators(validator_config, test)
    init_wandb(config, my_uid, wallet)
    loop = asyncio.get_event_loop()

    weight_setter = (WeightSetter if not test else TestWeightSetter)(
        loop, dendrite, subtensor, config, wallet, text_vali, image_vali, embed_vali)
    validator_app.weight_setter = weight_setter

    if run_aio_app:
        try:
            web.run_app(validator_app, port=config.http_port, loop=loop, shutdown_timeout=120)
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Exiting validator.")
        finally:
            state = utils.get_state()
            utils.save_state_to_file(state)
            if config.wandb_on:
                wandb.finish()


if __name__ == "__main__":
    main()
