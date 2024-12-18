from dotenv import load_dotenv
import argparse
import os
from pathlib import Path

import bittensor as bt

load_dotenv()  # Load environment variables from .env file


class Config:
    def __init__(self):
        super().__init__()

        self.ENV = os.getenv('ENV')

        self.WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
        self.AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        self.PIXABAY_API_KEY = os.getenv('PIXABAY_API_KEY')

        self.CORTEXT_MINER_ADDITIONAL_WHITELIST_VALIDATOR_KEYS = os.getenv(
            'CORTEXT_MINER_ADDITIONAL_WHITELIST_VALIDATOR_KEYS')
        self.RICH_TRACEBACK = os.getenv('RICH_TRACEBACK')

        self.WALLET_NAME = os.getenv('WALLET_NAME')
        self.HOT_KEY = os.getenv('HOT_KEY')
        self.NET_UID = int(os.getenv('NET_UID', 18))
        self.ASYNC_TIME_OUT = int(os.getenv('ASYNC_TIME_OUT', 60))
        self.AXON_PORT = int(os.getenv('AXON_PORT', 8098))
        self.EXTERNAL_IP = os.getenv('EXTERNAL_IP')

        self.BT_SUBTENSOR_NETWORK = 'test' if self.ENV == 'test' else 'finney'
        self.WANDB_OFF = False if self.ENV == 'prod' else True
        # still can use the --logging.debug and --logging.trace to turn on logging
        self.LOGGING_TRACE = False # if self.ENV == 'prod' else True
        self.BLACKLIST_AMT = 5000 if self.ENV == 'prod' else 0
        self.BLOCKS_PER_EPOCH = int(os.getenv('BLOCKS_PER_EPOCH', 100))
        self.WAIT_NEXT_BLOCK_TIME = int(os.getenv('WAIT_NEXT_BLOCK_TIME', 1))
        self.NO_SET_WEIGHTS = os.getenv('NO_SET_WEIGHTS', False)
        self.NO_SERVE = os.getenv('NO_SERVE', False)


    def __repr__(self):
        return (
            f"Config(BT_SUBTENSOR_NETWORK={self.BT_SUBTENSOR_NETWORK}, WALLET_NAME={self.WALLET_NAME}, HOT_KEY={self.HOT_KEY}"
            f", NET_UID={self.NET_UID}, WANDB_OFF={self.WANDB_OFF}, LOGGING_TRACE={self.LOGGING_TRACE}")


def get_config() -> (bt.config, Config):
    default_config = Config()
    parser = argparse.ArgumentParser()

    # remove_argument(parser, "--axon.port")
    parser.add_argument(
        "--axon.port", type=int, default=default_config.AXON_PORT, help="Port to run the axon on."
    )

    # remove_argument(parser, "--axon.external_ip")
    parser.add_argument(
        "--axon.external_ip", type=str, default=bt.utils.networking.get_external_ip(), help="IP for the metagraph"
    )

    # remove_argument(parser, "--wallet.name")
    parser.add_argument(
        "--wallet.name",
        default=default_config.WALLET_NAME,
        help="Bittensor Wallet Name to use."
    )

    # remove_argument(parser, "--wallet.hotkey")
    parser.add_argument(
        "--wallet.hotkey",
        default=default_config.HOT_KEY,
        help="Wallet Hotkey"
    )

    # Chain endpoint to connect to
    parser.add_argument(
        "--subtensor.chain_endpoint",
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="Chain endpoint to connect to.",
    )

    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=default_config.NET_UID, help="The chain subnet uid.")

    parser.add_argument(
        "--miner.root",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.bittensor/miners/",
    )

    parser.add_argument(
        "--miner.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="Bittensor Miner",
    )

    # Run config.
    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=int,
        help="Blocks until the miner sets weights on chain",
        default=default_config.BLOCKS_PER_EPOCH,
    )

    # Switches.
    parser.add_argument(
        "--miner.no_set_weights",
        action="store_true",
        help="If True, the miner does not set weights.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_serve",
        action="store_true",
        help="If True, the miner doesnt serve the axon.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_start_axon",
        action="store_true",
        help="If True, the miner doesnt start the axon.",
        default=False,
    )

    # Mocks.
    parser.add_argument(
        "--miner.mock_subtensor",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )

    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="If True, wandb is not turned on.",
        default=default_config.WANDB_OFF,
    )

    # remove_argument(parser, "--logging.trace")
    parser.add_argument(
        "--logging.trace",
        action="store_true",
        help="If True, logging tracing is turned on.",
        default=default_config.LOGGING_TRACE,
    )

    parser.add_argument(
        "--logging.debug",
        action="store_true",
        help="If True, logging tracing is turned on.",
        default=default_config.LOGGING_TRACE,
    )

    parser.add_argument(
        "--logging.info",
        action="store_true",
        help="If True, logging tracing is turned on.",
        default=default_config.LOGGING_TRACE,
    )

    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    bt_config = bt.config(parser)

    # Logging captures events for diagnosis or understanding miner's behavior.
    full_path = Path(f"{bt_config.logging.logging_dir}/{bt_config.wallet.name}/{bt_config.wallet.hotkey}"
                     f"/netuid{bt_config.netuid}/miner").expanduser()
    bt_config.full_path = str(full_path)
    # Ensure the directory for logging exists, else create one.
    full_path.mkdir(parents=True, exist_ok=True)

    bt.axon.check_config(bt_config)
    bt.logging.check_config(bt_config)

    if 'test' in bt_config.subtensor.chain_endpoint:
        bt_config.subtensor.network = 'test'
    else:
        bt_config.subtensor.network = 'finney'

    return bt_config, default_config


config = Config()
