import asyncio
import os
import sys

import base  # noqa
from validators.services.bittensor import bt_validator as bt
import cortext
import wandb
from cortext import utils
from validators.services.validators.embeddings_validator import EmbeddingsValidator
from validators.services.validators.image_validator import ImageValidator
from validators.services.validators.text_validator import TextValidator
from validators.weight_setter import WeightSetter

text_vali = None
image_vali = None
embed_vali = None
metagraph = None
wandb_runs = {}


def init_wandb(config, my_uid, wallet: bt.wallet):
    if not config.wandb_on:
        return

    run_name = f"validator-{my_uid}-{cortext.__version__}"
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = cortext.__version__
    config.type = "validator"

    # Initialize the wandb run for the single project
    run = wandb.init(
        name=run_name, project=cortext.PROJECT_NAME, entity="cortex-t", config=config, dir=config.full_path, reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{cortext.PROJECT_NAME}'")


def initialize_components(config: bt.config):
    global metagraph
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    dendrite = bt.dendrite(wallet=wallet)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Your validator: {wallet} is not registered to chain connection: "
            f"{subtensor}. Run btcli register --netuid 18 and try again."
        )
        sys.exit()
    return wallet, subtensor, dendrite, my_uid


def initialize_validators(vali_config, test=False):
    global text_vali, image_vali, embed_vali

    text_vali = TextValidator(**vali_config)
    image_vali = ImageValidator(**vali_config)
    embed_vali = EmbeddingsValidator(**vali_config)
    bt.logging.info("initialized_validators")


def main(test=False) -> None:
    validator_config = {"dendrite": dendrite, "config": config, "subtensor": subtensor, "wallet": wallet}
    initialize_validators(validator_config, test)
    init_wandb(config, my_uid, wallet)
    loop = asyncio.get_event_loop()
    weight_setter = WeightSetter(loop, dendrite, subtensor, config, wallet, text_vali, image_vali, embed_vali)
    state_path = os.path.join(config.full_path, "state.json")
    utils.get_state(state_path)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
    finally:
        state = utils.get_state(state_path)
        utils.save_state_to_file(state, state_path)
        if config.wandb_on:
            wandb.finish()


if __name__ == "__main__":
    main()
