import argparse
import time
import subprocess
import cortext
from cortext.utils import get_version, send_discord_alert

webhook_url = ""
current_version = cortext.__version__


def update_and_restart(pm2_name, netuid, wallet_name, wallet_hotkey, address, autoupdate, logging, wandb_on):
    global current_version
    wandb = "--wandb_on" if wandb_on else ""
    subprocess.run(["pm2", "start", "--name", pm2_name, f"python3 -m validators.validator --wallet.name {wallet_name}"
                                                        f" --wallet.hotkey {wallet_hotkey} "
                                                        f" --netuid {netuid} "
                                                        f"--subtensor.chain_endpoint {address} "
                                                        f"--logging.level {logging} {wandb}"])
    while True:
        latest_version = get_version()
        print(f"Current version: {current_version}")
        print(f"Latest version: {latest_version}")

        if current_version != latest_version and latest_version != None:
            if not autoupdate:
                send_discord_alert(
                    f"Your validator not running the latest code ({current_version}). You will quickly lose vtrust if you don't update to version {latest_version}",
                    webhook_url)
            print("Updating to the latest version...")
            subprocess.run(["pm2", "delete", pm2_name])
            subprocess.run(["git", "reset", "--hard"])
            subprocess.run(["git", "pull"])
            subprocess.run(["pip", "install", "-e", "."])
            subprocess.run(["pip", "install", "httpx==0.27.2"])
            subprocess.run(["pip", "uninstall", "uvloop"])
            subprocess.run(
                ["pm2", "start", "--name", pm2_name, f"python3 -m validators.validator --wallet.name {wallet_name}"
                                                     f" --wallet.hotkey {wallet_hotkey} "
                                                     f" --netuid {netuid} "
                                                     f"--subtensor.chain_endpoint {address} "
                                                     f"--logging.level {logging} {wandb}"])
            current_version = latest_version

        print("All up to date!")
        time.sleep(180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically update and restart the validator process when a new version is released."
    )

    parser.add_argument("--pm2_name", required=False, default="main-process", help="Name of the PM2 process.")
    parser.add_argument("--wallet_name", required=False, default="default", help="Name of the wallet.")
    parser.add_argument("--wallet_hotkey", required=False, default="default", help="Hotkey for the wallet.")
    parser.add_argument("--netuid", required=False, default=18, help="netuid for validator")
    parser.add_argument("--subtensor.chain_endpoint", required=False, default="wss://entrypoint-finney.opentensor.ai:443", dest="address")
    parser.add_argument("--autoupdate", action='store_true',  dest="autoupdate")
    parser.add_argument("--logging", required=False, default="info")
    parser.add_argument("--wandb_on", action='store_true', required=False, dest="wandb_on")
    parser.add_argument("--max_miners_cnt", type=int, default=30)

    args = parser.parse_args()

    try:
        update_and_restart(args.pm2_name, args.netuid, args.wallet_name, args.wallet_hotkey, args.address,
                           args.autoupdate, args.logging, args.wandb_on)
    except Exception as e:
        parser.error(f"An error occurred: {e}")