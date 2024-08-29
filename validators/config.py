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

        # bittensor config
        self.WALLET_NAME = os.getenv('WALLET_NAME')
        self.HOT_KEY = os.getenv('HOT_KEY')
        self.NET_UID = int(os.getenv('NET_UID', 0))
        self.AXON_PORT = int(os.getenv('AXON_PORT', 8000))
        self.BT_SUBTENSOR_NETWORK = 'test' if self.ENV == 'test' else 'finney'

        # logging config
        self.WANDB_OFF = False if self.ENV == 'prod' else True
        self.LOGGING_TRACE = False if self.ENV == 'prod' else True


def get_config() -> bt.config:
    default_config = Config()
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtensor.network", type=str, default=default_config.BT_SUBTENSOR_NETWORK)
    parser.add_argument("--wallet.name", type=str, default=default_config.WALLET_NAME)
    parser.add_argument("--wallet.hotkey", type=str, default=default_config.HOT_KEY)
    parser.add_argument("--netuid", type=int, default=default_config.NET_UID)
    parser.add_argument("--wandb_off", action="store_false", dest="wandb_on", default=default_config.WANDB_OFF)
    parser.add_argument("--axon.port", type=int, default=default_config.AXON_PORT)

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

    return bt_config_


app_config = Config()
bt_config = get_config()
