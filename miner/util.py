import bittensor as bt
import wandb
import time
import json
import traceback

from cortext.utils import get_version
import cortext

valid_hotkeys = []


def get_valid_hotkeys(config):
    global valid_hotkeys
    api = wandb.Api()
    subtensor = bt.subtensor(config=config)
    while True:
        metagraph = subtensor.metagraph(18)
        try:
            runs = api.runs(f"cortex-t/{cortext.PROJECT_NAME}")
            latest_version = get_version()
            for run in runs:
                if run.state == "running":
                    try:
                        # Extract hotkey and signature from the run's configuration
                        hotkey = run.config['hotkey']
                        signature = run.config['signature']
                        version = run.config['version']
                        bt.logging.debug(f"found running run of hotkey {hotkey}, {version} ")

                        if latest_version is None:
                            bt.logging.error("Github API call failed!")
                            continue

                        if latest_version not in (version, None):
                            bt.logging.debug(
                                f"Version Mismatch: Run version {version} does not match GitHub version {latest_version}"
                            )
                            continue

                        # Check if the hotkey is registered in the metagraph
                        if hotkey not in metagraph.hotkeys:
                            bt.logging.debug(f"Invalid running run: The hotkey: {hotkey} is not in the metagraph.")
                            continue

                        # Verify the signature using the hotkey
                        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
                            bt.logging.debug(f"Failed Signature: The signature: {signature} is not valid")
                            continue

                        if hotkey not in valid_hotkeys:
                            valid_hotkeys.append(hotkey)
                    except Exception:
                        bt.logging.debug(f"exception in get_valid_hotkeys: {traceback.format_exc()}")

            bt.logging.info(f"total valid hotkeys list = {valid_hotkeys}")
            time.sleep(180)

        except json.JSONDecodeError as e:
            bt.logging.debug(f"JSON decoding error: {e} {run.id}")
