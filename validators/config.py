import bittensor as bt

from dotenv import load_dotenv
import argparse
import os
from pathlib import Path

load_dotenv()  # Load environment variables from .env file


class Config:
    def __init__(self):
        super().__init__()

        self.ENV = os.getenv('ENV')
        self.ASYNC_TIME_OUT = int(os.getenv('ASYNC_TIME_OUT', 60))
        self.BT_SUBTENSOR_NETWORK = 'test' if self.ENV == 'test' else 'finney'

    @staticmethod
    def check_required_env_vars():
        AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
        AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        if all([AWS_SECRET_KEY, AWS_ACCESS_KEY]):
            pass
        else:
            bt.logging.info("AWS_KEY is not provided correctly. so exit system")
            exit(0)


def get_config() -> bt.config:
    Config.check_required_env_vars()
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtensor.chain_endpoint", type=str)
    parser.add_argument("--wallet.name", type=str)
    parser.add_argument("--wallet.hotkey", type=str)
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--wandb_off", action="store_true", dest="wandb_off")
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument('--logging.info', action='store_true')
    parser.add_argument('--logging.debug', action='store_true')
    parser.add_argument('--logging.trace', action='store_true')

    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    bt_config_ = bt.config(parser)
    bt.configs.append(bt_config_)
    bt_config_ = bt.config.merge_all(bt.configs)

    # Logging captures events for diagnosis or understanding miner's behavior.
    full_path = Path(f"{bt_config_.logging.logging_dir}/{bt_config_.wallet.name}/{bt_config_.wallet.hotkey}"
                     f"/netuid{bt_config_.netuid}/miner").expanduser()
    bt_config_.full_path = str(full_path)
    # Ensure the directory for logging exists, else create one.
    full_path.mkdir(parents=True, exist_ok=True)

    bt.axon.check_config(bt_config_)
    bt.logging.check_config(bt_config_)
    if 'test' in bt_config_.subtensor.chain_endpoint:
        bt_config_.subtensor.network = 'test'
    else:
        bt_config_.subtensor.network = 'finney'

    bt.logging.info(bt_config_)
    return bt_config_


app_config = Config()
bt_config = get_config()
