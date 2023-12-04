import time
import template
import requests
import subprocess

from template.utils import get_version


pm2_name = "net18vali"
wallet_name = "1"
wallet_hotkey = "1"

subprocess.run(["pm2", "start", "validator.py", "--interpreter", "python3", "--name", pm2_name, "--", "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey, "--netuid", "24", "--subtensor.network", "test"])

def update_and_restart():
    current_version = template.__version__
    latest_version = get_version()
    print(f"version is {current_version}")
    print(f"latest version is {latest_version}")

    if current_version != latest_version:
        print("Updating to the latest version...")
        subprocess.run(["pm2", "delete", pm2_name])
        subprocess.run(["git", "pull"])
        subprocess.run(["pip", "install", "-e", "."])
        subprocess.run(["pm2", "start", "validator.py", "--interpreter", "python3", "--name", pm2_name, "--", "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey, "--netuid", "24", "--subtensor.network", "test"])

        time.sleep(10)

if __name__ == "__main__":
    update_and_restart()