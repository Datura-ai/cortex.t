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
        bt_config = bt.config(parser)
        """Load configuration from environment variables."""
        return Config(
            wallet_name=bt_config.wallet.name,  # Default to an empty string if not set
            wallet_hotkey=bt_config.wallet.hotkey,
            api_key=os.getenv("CURSOR_API_KEY", "")
        )


# Load config
config = Config.from_env()

