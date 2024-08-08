import argparse
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
env_ = str(os.getenv('ENV')).upper()
default_address = "wss://test.finney.opentensor.ai:443" if env_ == 'DEV' \
    else "wss://bittensor-finney.api.onfinality.io/public-ws"
subtensor_network = 'test' if env_ == 'DEV' else 'local'
net_uid = '196' if env_ == 'DEV' else '18'


def update_and_restart(wallet_name, wallet_hotkey, address, autoupdate):
    subprocess.run(["python3", "-m", "validators.validator",
                    "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey, "--netuid", net_uid,
                    "--subtensor.network", subtensor_network, "--subtensor.chain_endpoint", address])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically update and restart the validator process when a new version is released.",
        epilog="Example usage: python start_validator.py --pm2_name 'net18vali' --wallet_name 'wallet1' --wallet_hotkey 'key123' [--address 'wss://...'] [--no-autoupdate]"
    )

    parser.add_argument("--wallet_name", required=True, help="Name of the wallet.")
    parser.add_argument("--wallet_hotkey", required=True, help="Hotkey for the wallet.")
    parser.add_argument("--address", default=default_address,
                        help="Subtensor chain_endpoint, defaults to 'wss://bittensor-finney.api.onfinality.io/public-ws' if not provided.")
    parser.add_argument("--no-autoupdate", action='store_false', dest='autoupdate',
                        help="Disable automatic update. Only send a Discord alert. Add your webhook at the top of the script.")

    args = parser.parse_args()

    try:
        update_and_restart(args.wallet_name, args.wallet_hotkey, args.address, args.autoupdate)
    except Exception as e:
        parser.error(f"An error occurred: {e}")
