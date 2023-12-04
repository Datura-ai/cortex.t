import os
import time
import wandb
import traceback
import template
import argparse
import bittensor as bt
from template.utils import get_version

valid_hotkeys = []

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

def get_valid_hotkeys(config):
    global valid_hotkeys
    api = wandb.Api()
    subtensor = bt.subtensor(config=config)
    while True:
        metagraph = subtensor.metagraph(18)
        runs = api.runs(f"cortex-t/{template.PROJECT_NAME}")
        latest_version = get_version()
        for run in runs:
            if run.state == "running":
                try:
                    # Extract hotkey and signature from the run's configuration
                    hotkey = run.config['hotkey']
                    signature = run.config['signature']
                    version = run.config['version']

                    bt.logging.info(f"hotkey is running {version}")
                    if latest_version != None and version != latest_version:
                        print(f'Version Mismatch: Run version {version} does not match GitHub version {latest_version}')
                        continue
                    
                    bt.logging.info("version matches or github api failed")

                    # Check if the hotkey is registered in the metagraph
                    if hotkey not in metagraph.hotkeys:
                        print(f'Invalid running run: The hotkey: {hotkey} is not in the metagraph.')
                        continue

                    # Verify the signature using the hotkey
                    if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
                        print(f'Failed Signature: The signature: {signature} is not valid')
                        continue

                    if hotkey not in valid_hotkeys:
                        valid_hotkeys.append(hotkey)
                except Exception as e:
                    bt.logging.error(f"exception in get_valid_hotkeys: {traceback.format_exc()}")

        bt.logging.info(f"total valid hotkeys list = {valid_hotkeys}")
        time.sleep(10)
        return valid_hotkeys

config = get_config()
valid_hotkeys = get_valid_hotkeys(config)
print(valid_hotkeys)