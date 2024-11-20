import dataclasses
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    wallet_name: str
    wallet_hotkey: str

    @staticmethod
    def from_env() -> "Config":
        """Load configuration from environment variables."""
        return Config(
            wallet_name=os.getenv("WALLET_NAME", "default"),  # Default to an empty string if not set
            wallet_hotkey=os.getenv("HOT_KEY", "default")
        )


# Load config
config = Config.from_env()
