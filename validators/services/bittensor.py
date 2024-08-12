from validators.config import bt_config
import bittensor as bt
import sys


class BittensorValidator:
    def __init__(self):
        self.config = bt_config
        bt.logging(config=config, logging_dir=config.full_path)
        self.logging = bt.logging
        self.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.check_wallet_registered_in_network()

    def check_wallet_registered_in_network(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: "
                f"{self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            sys.exit()


bt_validator = BittensorValidator()
