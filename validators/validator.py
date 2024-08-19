import asyncio
import os
import sys

import base  # noqa
from validators.services.bittensor import bt_validator as bt
import cortext
import wandb
from cortext import utils
from validators.weight_setter import WeightSetter
from validators.config import bt_config, app_config
from validators.services.bittensor import bt_validator as bt


def init_wandb():
    if not bt_config.wandb_on:
        return

    run_name = f"validator-{bt.my_uid}-{cortext.__version__}"
    bt_config.uid = bt.my_uid
    bt_config.hotkey = bt.wallet.hotkey.ss58_address
    bt_config.run_name = run_name
    bt_config.version = cortext.__version__
    bt_config.type = "validator"

    # Initialize the wandb run for the single project
    run = wandb.init(
        name=run_name, project=cortext.PROJECT_NAME, entity="cortex-t", config=bt_config, dir=bt_config.full_path,
        reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = bt.wallet.hotkey.sign(run.id.encode()).hex()
    bt_config.signature = signature
    wandb.config.update(bt_config, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{cortext.PROJECT_NAME}'")

def main(test=False) -> None:
    init_wandb()
    loop = asyncio.get_event_loop()
    weight_setter = WeightSetter(loop)
    state_path = os.path.join(bt_config.full_path, "state.json")
    utils.get_state(state_path)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
    finally:
        state = utils.get_state(state_path)
        utils.save_state_to_file(state, state_path)
        if bt_config.wandb_on:
            wandb.finish()


if __name__ == "__main__":
    main()
