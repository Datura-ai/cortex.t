import os
import time
import argparse
import traceback
import bittensor as bt
from typing import Tuple, List, Generator, AsyncGenerator
import template
import openai
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from starlette.types import Send
from functools import partial

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", default="my_custom_value")
    parser.add_argument("--netuid", type=int, default=1)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/miner")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

class Miner:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.wallet, self.subtensor, self.metagraph, self.my_subnet_uid = self.initialize_wallet()
        self.axon = self.initialize_axon()

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")

    def initialize_wallet(self):
        wallet = bt.wallet(config=self.config)
        subtensor = bt.subtensor(config=self.config)
        metagraph = subtensor.metagraph(self.config.netuid)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(f"Your miner: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again.")
            exit()
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        return wallet, subtensor, metagraph, my_subnet_uid

    def initialize_axon(self):
        axon = bt.axon(wallet=self.wallet, config=self.config)
        axon.attach(
            forward_fn=self.process_question,
            blacklist_fn=self.blacklist_fn,
            priority_fn=self.priority_fn
        )
        axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        axon.start()
        return axon

    def blacklist_fn(self, synapse: template.protocol.StreamPrompting) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
        return False, "Hotkey recognized!"

    def priority_fn(self, synapse: template.protocol.StreamPrompting) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def process_question(self, synapse: template.protocol.StreamPrompting) -> template.protocol.StreamPrompting:
        bt.logging.info(f"Received synapse: {synapse}")
        prompt = synapse.messages[0]  # If messages is a list of strings
        engine = synapse.engine
        bt.logging.info(f"Processing query from validator: '{prompt}' using {engine}")

        async def _prompt(openai_stream, send: Send):
            async for message in openai_stream:
                if 'choices' in message and message['choices']:
                    chunk_message = message['choices'][0]['message']['content']
                    bt.logging.debug(f"Sending chunk: {chunk_message}")
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk_message.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.trace(f"Streamed chunk: {chunk_message}")

            await send({"type": "http.response.body", "body": b"", "more_body": False})

        response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            stream=True
        )
        return synapse.completion(partial(_prompt, response))

    def run(self):
        step = 0
        while True:
            try:
                if step % 5 == 0:
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                step += 1
                time.sleep(1)
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception:
                bt.logging.error(traceback.format_exc())

if __name__ == "__main__":
    miner = Miner(get_config())
    miner.run()
