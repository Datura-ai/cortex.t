import os
import wandb
import traceback
import template
import argparse
import bittensor as bt

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
    api = wandb.Api()
    valid_hotkeys = []
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(18)
    runs = api.runs(f"{template.PROJECT_NAME}")

    for run in runs:
        if run.state == "running":
            try:
                # Extract hotkey and signature from the run's configuration
                hotkey = run.config['hotkey']
                signature = run.config['signature']

                # Check if the hotkey is registered in the metagraph
                if hotkey not in metagraph.hotkeys:
                    print(f'Invalid running run: The hotkey: {hotkey} is not in the metagraph.')
                    continue

                # Verify the signature using the hotkey
                if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
                    print(f'Failed Signature: The signature: {signature} is not valid')
                    continue

                valid_hotkeys.append(hotkey)
            except Exception as e:
                bt.logging.error(f"exception in get_valid_hotkeys: {traceback.format_exc()}")

    return valid_hotkeys


config = get_config()
valid_hotkeys = get_valid_hotkeys(config)
print(valid_hotkeys)