import os
import random
import time
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
import bittensor as bt

import wandb
import cortext
from cortext import utils, dendrite
from validators.weight_setter import WeightSetter
from validators.services.cache import cache_service

# Load environment variables from .env file
load_dotenv()
random.seed(time.time())


class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def get(self, key, default=None):
        if '.' in key:
            group, key = key.split('.', 1)
            return getattr(self, group, NestedNamespace()).get(key, default)
        return self.__dict__.get(key, default)


class Config:
    def __init__(self, args):

        # Add command-line arguments to the Config object
        for key, value in vars(args).items():
            setattr(self, key, value)

    @staticmethod
    def check_required_env_vars():
        required_vars = ['AWS_ACCESS_KEY', 'AWS_SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            bt.logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            exit(1)

    def get(self, key, default=None):
        return getattr(self, key, default)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validator Configuration")
    parser.add_argument("--subtensor.chain_endpoint", type=str,
                        default="wss://entrypoint-finney.opentensor.ai:443")  # for testnet: wss://test.finney.opentensor.ai:443
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument("--wandb_on", action="store_true")
    parser.add_argument("--max_miners_cnt", type=int, default=30)
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument("--logging.level", choices=['info', 'debug', 'trace'], default='info')
    parser.add_argument("--autoupdate", action="store_true", help="Enable auto-updates")
    parser.add_argument("--image_validator_probability", type=float, default=0.001)
    parser.add_argument("--async_time_out", type=int, default=60)
    bt_config = bt.config(parser)
    return bt_config


def setup_logging(config):
    if config.logging.level == 'trace':
        bt.logging.set_trace()
    elif config.logging.level == 'debug':
        bt.logging.set_debug()
    else:
        # set to info by default
        pass
    bt.logging.info(f"Set logging level to {config.logging.level}")

    full_path = Path(
        f"~/.bittensor/validators/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator").expanduser()
    full_path.mkdir(parents=True, exist_ok=True)
    config.full_path = str(full_path)

    bt.logging.info(f"Arguments: {vars(config)}")


def init_wandb(config):
    if not config.wandb_on:
        return

    run_name = f"validator-{config.wallet.hotkey.ss58_address}-{cortext.__version__}"
    config.run_name = run_name
    config.version = cortext.__version__
    config.type = "validator"

    run = wandb.init(
        name=run_name,
        project=cortext.PROJECT_NAME,
        entity="cortex-t",
        config=config.__dict__,
        dir=config.full_path,
        reinit=True
    )

    signature = config.wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config.__dict__, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{cortext.PROJECT_NAME}'")


async def close_all_connections():
    tasks = []
    for key, connection in dendrite.CortexDendrite.miner_to_session.items():
        tasks.append(connection.close())
    await asyncio.gather(*tasks)


def main():
    Config.check_required_env_vars()
    config = parse_arguments()
    # config = Config(args)

    setup_logging(config)

    config.wallet = bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey)
    config.dendrite = dendrite.CortexDendrite(wallet=config.wallet)

    bt.logging.info(f"Config: {vars(config)}")

    init_wandb(config)
    loop = asyncio.get_event_loop()
    weight_setter = WeightSetter(config=config, cache=cache_service, loop=loop)
    state_path = os.path.join(config.full_path, "state.json")
    utils.get_state(state_path)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
    finally:
        bt.logging.info("stopping axon server.")
        # bt.logging.info(
        #     f"closing all sessins. total connections is {len(dendrite.CortexDendrite.miner_to_session.keys())}")
        # asyncio.run(close_all_connections())
        weight_setter.axon.stop()
        bt.logging.info("updating status before exiting validator")
        state = utils.get_state(state_path)
        utils.save_state_to_file(state, state_path)
        bt.logging.info("closing connection of cache database.")
        cache_service.close()
        if config.wandb_on:
            wandb.finish()


if __name__ == "__main__":
    main()
