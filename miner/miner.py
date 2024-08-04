import argparse
import asyncio
import base64
import aioboto3
import copy
import json
import os
import pathlib
import httpx
import threading
import time
import traceback
from collections import deque
from functools import partial
from typing import Tuple

import bittensor as bt
import google.generativeai as genai
import wandb
from config import check_config, get_config
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from anthropic_bedrock import AsyncAnthropicBedrock

import cortext
from cortext.protocol import Embeddings, ImageResponse, IsAlive, StreamPrompting, TextPrompting
from cortext.utils import get_version, get_api_key
import sys

from starlette.types import Send
from miner.config import config
from pathlib import Path

valid_hotkeys = []


class StreamMiner():
    def __init__(self, axon=None, wallet=None, subtensor=None):

        self.last_epoch_block = None
        self.my_subnet_uid = None
        self.axon = axon
        self.wallet = wallet
        self.subtensor = subtensor

        bt.logging.info("starting stream miner")

        self.init_bittensor()
        self.init_axon()

        # Instantiate runners
        self.prompt_cache: dict[str, Tuple[str, int]] = {}
        self.request_timestamps: dict = {}
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def init_bittensor(self):
        # Activating Bittensor's logging with the set configurations.
        bt.logging(trace=config.LOGGING_TRACE)
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = self.wallet or bt.wallet(name=config.WALLET_NAME, hotkey=config.HOT_KEY)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        self.subtensor = self.subtensor or bt.subtensor(network=config.BT_SUBTENSOR_NETWORK)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {config.NET_UID} "
            f"on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(config.NET_UID)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.check_hotkey_validation()

    def init_axon(self):

        bt.logging.debug(
            f"Starting axon on port {config.AXON_PORT} and external ip {config.EXTERNAL_IP}"
        )
        self.axon = self.axon or bt.axon(
            wallet=self.wallet,
            port=config.AXON_PORT,
            external_ip=config.EXTERNAL_IP,
        )

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")
        print(f"Attaching forward function to axon. {self.prompt}")

        axon_bridges = [(self.prompt, self.blacklist_prompt), (self.is_alive, self.blacklist_is_alive),
                        (self.images, self.blacklist_images), (self.embeddings, self.blacklist_embeddings),
                        (self.text, None)]

        for forward_fn, blacklist_fn in axon_bridges:
            self.axon = self.axon.attach(forward_fn=forward_fn, blacklist_fn=blacklist_fn)

        bt.logging.info(f"Axon created: {self.axon}")

    def check_hotkey_validation(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered to this subnet"
                f"\nRun btcli recycle_register --netuid 18 and try again. "
            )
            sys.exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def text(self, synapse: TextPrompting) -> TextPrompting:
        synapse.completion = "completed by miner"
        return synapse

    def base_blacklist(self, synapse, blacklist_amt=5000) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            uid = None
            for _uid, _axon in enumerate(self.metagraph.axons):  # noqa: B007
                if _axon.hotkey == hotkey:
                    uid = _uid
                    break

            if uid is None and cortext.ALLOW_NON_REGISTERED is False:
                return True, f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}"

            # check the stake
            stake = self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]
            if stake < blacklist_amt:
                return True, f"Blacklisted a low stake {synapse_type} request: {stake} < {blacklist_amt} from {hotkey}"

            time_window = cortext.MIN_REQUEST_PERIOD * 60
            current_time = time.time()

            if hotkey not in self.request_timestamps:
                self.request_timestamps[hotkey] = deque()

            # Remove timestamps outside the current time window
            while self.request_timestamps[hotkey] and current_time - self.request_timestamps[hotkey][0] > time_window:
                self.request_timestamps[hotkey].popleft()

            # Check if the number of requests exceeds the limit
            if len(self.request_timestamps[hotkey]) >= cortext.MAX_REQUESTS:
                return (
                    True,
                    f"Request frequency for {hotkey} exceeded: "
                    f"{len(self.request_timestamps[hotkey])} requests in {cortext.MIN_REQUEST_PERIOD} minutes. "
                    f"Limit is {cortext.MAX_REQUESTS} requests."
                )

            self.request_timestamps[hotkey].append(current_time)

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def blacklist_prompt(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.PROMPT_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_is_alive(self, synapse: IsAlive) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.ISALIVE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_images(self, synapse: ImageResponse) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.IMAGE_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_embeddings(self, synapse: Embeddings) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.EMBEDDING_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def run(self):
        bt.logging.info(
            f"Serving axon {StreamPrompting} "
            f"on network: {self.subtensor.chain_endpoint} "
            f"with netuid: {config.NET_UID}"
        )
        self.axon.serve(config.NET_UID, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {config.AXON_PORT}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info("Starting main loop")
        step = 0
        try:
            while not self.should_exit:
                _start_epoch = time.time()
                # --- Wait until next epoch.
                current_block = self.subtensor.get_current_block()
                while (
                        current_block - self.last_epoch_block
                        < config.BLOCKS_PER_EPOCH
                ):
                    # --- Wait for next block.
                    time.sleep(config.WAIT_NEXT_BLOCK_TIME)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=config.NET_UID,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                    f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid]} | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                # --- Set weights.
                if not config.NO_SET_WEIGHTS:
                    pass
                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            sys.exit()

        except Exception:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self) -> None:
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self) -> None:
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.debug("answered to be active")
        synapse.completion = "True"
        return synapse


if __name__ == "__main__":
    with StreamMiner():
        while True:
            time.sleep(1)
