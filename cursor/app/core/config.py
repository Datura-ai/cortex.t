import dataclasses
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import argparse
import bittensor as bt

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    wallet_name: str
    wallet_hotkey: str
    api_key: str

    @staticmethod
    def from_env() -> "Config":
        parser = argparse.ArgumentParser(description="Bittensor example script")
        parser.add_argument('--wallet.name', type=str, default="default", help="wallet name")
        parser.add_argument('--wallet.hotkey', type=str, default="default", help="wallet hotkey")
        parser.add_argument('--netuid', type=int, default=18, help="netuid")
        parser.add_argument('--subtensor.chain_endpoint', type=str, default="finney", help="network name")
        bt_config = bt.config(parser)
        """Load configuration from environment variables."""
        return Config(
            wallet_name=bt_config.wallet.name,  # Default to an empty string if not set
            wallet_hotkey=bt_config.wallet.hotkey,
            api_key=os.getenv("CURSOR_API_KEY", ""),
            netuid = bt_config.netuid,
            network = "test" if "test" in bt_config.subtensor.chain_endpoint else "finney"
        )


# Load config
config = Config.from_env()

