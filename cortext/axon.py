import bittensor
import bittensor as bt
from substrateinterface import Keypair
from bittensor.core.errors import SynapseDendriteNoneException


class CortexAxon(bt.axon):
    def default_verify(self, synapse: bittensor.Synapse):
        if synapse.dendrite is not None:
            keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

            # Build the signature messages.
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

            # Build the unique endpoint key.
            endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

            if not keypair.verify(message, synapse.dendrite.signature):
                raise Exception(
                    f"Signature mismatch with {message} and {synapse.dendrite.signature}"
                )

            # Success
            self.nonces[endpoint_key] = synapse.dendrite.nonce  # type: ignore
        else:
            raise SynapseDendriteNoneException()
