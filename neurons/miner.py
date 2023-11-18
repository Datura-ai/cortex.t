import copy
import time
import asyncio
import argparse
import threading
import traceback
import os
from abc import ABC, abstractmethod
from functools import partial
from starlette.types import Send
from openai import OpenAI
from openai import AsyncOpenAI
import bittensor as bt
from transformers import GPT2Tokenizer
from typing import List, Dict, Tuple, Union, Callable, Awaitable
from template.protocol import StreamPrompting, IsAlive, ImageResponse
from config import get_config, check_config

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=30.0)


class StreamMiner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        bt.logging.info("starting stream miner")
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)  # TODO: duplicate print?
        self.prompt_cache: Dict[str, Tuple[str, int]] = {}

        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # The axon handles request processing, allowing validators to send this process requests.
        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)
        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")
        print(f"Attaching forward function to axon. {self._prompt}")
        self.axon.attach(
            forward_fn=self._prompt,
        ).attach(
            forward_fn=self.is_alive,
        ).attach(
            forward_fn=self._images,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: Dict = {}

    @abstractmethod
    def config(self) -> "bt.Config":
        ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...

    def _prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        return self.prompt(synapse)

    async def _images(self, synapse: ImageResponse) -> ImageResponse:
        return await self.images(synapse)

    def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")
        synapse.completion = "True"
        return synapse

    @abstractmethod
    def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        ...

    @abstractmethod
    def images(self, synapse: ImageResponse) -> ImageResponse:
        ...

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            exit()
        bt.logging.info(
            f"Serving axon {StreamPrompting} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info(f"Starting main loop")
        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()

                # --- Wait until next epoch.
                current_block = self.subtensor.get_current_block()
                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):
                    # --- Wait for next bloc.
                    time.sleep(1)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                    f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                # --- Set weights.
                if not self.config.miner.no_set_weights:
                    pass
                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
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


class StreamingTemplateMiner(StreamMiner):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    async def images(self, synapse: ImageResponse) -> ImageResponse:
        bt.logging.info(f"called image axon {synapse}")
        try:
            # Extract necessary information from synapse
            engine = synapse.engine
            messages = synapse.messages
            size = synapse.size
            quality = synapse.quality
            style = synapse.style

            # Await the response from the asynchronous function
            meta = await client.images.generate(
                model=engine,
                prompt=messages,
                size=size,
                quality=quality,
                style=style,
                )

            image_created = meta.created
            image_url = meta.data[0].url
            image_revised_prompt = meta.data[0].revised_prompt
            # image_b64 = meta.data[0].revised_prompt

            image_data = {
                "created_at": image_created,
                "url": image_url,
                "revised_prompt": image_revised_prompt,
                # "b64": image_b64
            }

            synapse.completion = image_data
            bt.logging.info(f"returning image response of {synapse.completion}")
            return synapse

        except Exception as e:
            bt.logging.error(f"error in images: {e}\n{traceback.format_exc()}")



    def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        bt.logging.info(f"starting processing for synapse {synapse}")
        
        async def _prompt(synapse, send: Send):
            try:
                engine = synapse.engine
                messages = synapse.messages
                seed=synapse.seed
                bt.logging.info(synapse)
                bt.logging.info(f"question is {messages} with engine {engine}, seed: {seed}")
                response = await client.chat.completions.create(
                    model= engine,
                    messages= messages,
                    temperature= 0.0001,
                    stream= True,
                    seed=seed,
                )
                buffer = []
                N=1
                async for chunk in response:
                    token = chunk.choices[0].delta.content or ""
                    buffer.append(token)
                    if len(buffer) == N:
                        joined_buffer = "".join(buffer)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        bt.logging.info(f"Streamed tokens: {joined_buffer}")
                        buffer = []

                if buffer:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )
                    bt.logging.info(f"Streamed tokens: {joined_buffer}")
                    print(f"response is {response}")
            except Exception as e:
                bt.logging.error(f"error in _prompt {e}\n{traceback.format_exc()}")

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)


if __name__ == "__main__":
    with StreamingTemplateMiner():
        while True:
            time.sleep(1)
