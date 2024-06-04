import argparse
import asyncio
import base64
import aioboto3
import copy
import json
import os
import pathlib
import requests
import threading
import time
import traceback
import anthropic
from collections import deque
from functools import partial
from typing import Tuple

import bittensor as bt
import google.generativeai as genai
import wandb
from PIL import Image
from stability_sdk import client
from config import check_config, get_config
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from huggingface_hub import AsyncInferenceClient
from stability_sdk import client as stability_client
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from anthropic_bedrock import AsyncAnthropicBedrock, HUMAN_PROMPT, AI_PROMPT, AnthropicBedrock

import cortext
from cortext.protocol import Embeddings, ImageResponse, IsAlive, StreamPrompting, TextPrompting
from cortext.utils import get_version, get_api_key
import sys

from starlette.types import Send


# Set up api keys from .env file and initialze clients

# OpenAI
OpenAI.api_key = get_api_key("OpenAI", "OPENAI_API_KEY")
client = AsyncOpenAI(timeout=60.0)

# Stability API
# stability_key = get_api_key("Stability", "STABILITY_API_KEY")
# stability_api = stability_client.StabilityInference(key=stability_key, verbose=True)

# Anthropic
# Only if using the official claude for access instead of aws bedrock
claude_key = get_api_key("Anthropic", "ANTHROPIC_API_KEY")
claude_client = AsyncAnthropic()
claude_client.api_key = claude_key

# For AWS bedrock (default)
bedrock_client = AsyncAnthropicBedrock(
    # default is 10 minutes
    # more granular timeout options:  timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
    timeout=60.0,
)

# AWS Bedrock
bedrock_client_parameters = {
    "service_name": 'bedrock-runtime',
    "aws_access_key_id": get_api_key("AWS Bedrock", "AWS_ACCESS_KEY"),
    "aws_secret_access_key": get_api_key("AWS Bedrock", "AWS_SECRET_KEY"),
    "region_name": "us-east-1"
}

# For google/gemini
google_key=get_api_key("Google", "GOOGLE_API_KEY")
genai.configure(api_key=google_key)

# Groq
groq_key = get_api_key("Groq", "GROQ_API_KEY")
groq_client = AsyncGroq()
groq_client.api_key = groq_key

# Hugging Face
hugging_face_key = get_api_key("Hugging Face", "HUGGING_FACE_API_KEY")
hugging_face_client = AsyncInferenceClient(token=hugging_face_key)


# Wandb
netrc_path = pathlib.Path.home() / ".netrc"
wandb_api_key = os.getenv("WANDB_API_KEY")
bt.logging.info("WANDB_API_KEY is set")
bt.logging.info("~/.netrc exists:", netrc_path.exists())

if not wandb_api_key and not netrc_path.exists():
    raise ValueError("Please log in to wandb using `wandb login` or set the WANDB_API_KEY environment variable.")

valid_hotkeys = []

class StreamMiner():
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        bt.logging.info("starting stream miner")
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)
        self.prompt_cache: dict[str, Tuple[str, int]] = {}
        self.request_timestamps = {}

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
            f"Running miner for subnet: {self.config.netuid} "
            f"on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

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

        # The axon handles request processing, allowing validators to send this process requests.
        if axon is not None:
            self.axon = axon
        elif self.config.axon.external_ip is not None:
            bt.logging.debug(
                f"Starting axon on port {self.config.axon.port} and external ip {self.config.axon.external_ip}"
            )
            self.axon = bt.axon(
                wallet=self.wallet,
                port=self.config.axon.port,
                external_ip=self.config.axon.external_ip,
            )
        else:
            bt.logging.debug(f"Starting axon on port {self.config.axon.port}")
            self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")
        print(f"Attaching forward function to axon. {self.prompt}")
        self.axon.attach(
            forward_fn=self.prompt,
            blacklist_fn=self.blacklist_prompt,
        ).attach(
            forward_fn=self.is_alive,
            blacklist_fn=self.blacklist_is_alive,
        ).attach(
            forward_fn=self.images,
            blacklist_fn=self.blacklist_images,
        ).attach(
            forward_fn=self.embeddings,
            blacklist_fn=self.blacklist_embeddings,
        ).attach(
            forward_fn=self.text,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: dict = {}
        thread = threading.Thread(target=get_valid_hotkeys, args=(self.config,))
        thread.start()

    def text(self, synapse: TextPrompting) -> TextPrompting:
        synapse.completion = "completed by miner"
        return synapse

    def config(self) -> bt.config:
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        return bt.config(parser)

    def base_blacklist(self, synapse, blacklist_amt = 20000) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            # if hotkey in cortext.WHITELISTED_KEYS:
            #     return False,  f"accepting {synapse_type} request from {hotkey}"

            if hotkey not in valid_hotkeys:
                return True, f"Blacklisted a {synapse_type} request from a non-valid hotkey: {hotkey}"

            uid = None
            for uid, _axon in enumerate(self.metagraph.axons):  # noqa: B007
                if _axon.hotkey == hotkey:
                    break

            if uid is None and cortext.ALLOW_NON_REGISTERED is False:
                return True, f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}"

            # check the stake
            tao = self.metagraph.neurons[uid].S
            # metagraph.neurons[uid].S
            if tao < blacklist_amt:
                return True, f"Blacklisted a low stake {synapse_type} request: {tao} < {blacklist_amt} from {hotkey}"

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


    def blacklist_prompt( self, synapse: StreamPrompting ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.PROMPT_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_is_alive( self, synapse: IsAlive ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.ISALIVE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_images( self, synapse: ImageResponse ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.IMAGE_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_embeddings( self, synapse: Embeddings ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.EMBEDDING_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            sys.exit()
        bt.logging.info(
            f"Serving axon {StreamPrompting} "
            f"on network: {self.config.subtensor.chain_endpoint} "
            f"with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
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
                    < self.config.miner.blocks_per_epoch
                ):
                    # --- Wait for next block.
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

    async def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        bt.logging.info(f"started processing for synapse {synapse}")

        async def _prompt(synapse, send: Send):
            try:
                provider = synapse.provider
                model = synapse.model
                messages = synapse.messages
                seed = synapse.seed
                temperature = synapse.temperature
                max_tokens = synapse.max_tokens
                top_p = synapse.top_p
                top_k = synapse.top_k


                if provider == "OpenAI":
                    # Test seeds + higher temperature
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                        seed=seed,
                        max_tokens=max_tokens
                    )
                    buffer = []
                    n = 1
                    async for chunk in response:
                        token = chunk.choices[0].delta.content or ""
                        buffer.append(token)
                        if len(buffer) == n:
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

                elif provider == "Anthropic":
                    stream = await bedrock_client.completions.create(
                        prompt=f"\n\nHuman: {messages}\n\nAssistant:",
                        max_tokens_to_sample=max_tokens,
                        temperature=temperature,  # must be <= 1.0
                        top_k=top_k,
                        top_p=top_p,
                        model=model,
                        stream=True,
                    )

                    async for completion in stream:
                        if completion.completion:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": completion.completion.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {completion.completion}")

                    # Send final message to close the stream
                    await send({"type": "http.response.body", "body": b'', "more_body": False})

                elif provider == "Claude":
                    system_prompt = None
                    filtered_messages = []
                    for message in messages:
                        if message["role"] == "system":
                            system_prompt = message["content"]
                        else:
                            filtered_messages.append(message)

                    stream_kwargs = {
                        "max_tokens": max_tokens,
                        "messages": filtered_messages,
                        "model": model,
                    }

                    if system_prompt:
                        stream_kwargs["system"] = system_prompt

                    completion = claude_client.messages.stream(**stream_kwargs)
                    async with completion as stream:
                        async for text in stream.text_stream:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": text.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {text}")

                    # Send final message to close the stream
                    await send({"type": "http.response.body", "body": b'', "more_body": False})

                elif provider == "Gemini":
                    model = genai.GenerativeModel(model)
                    stream = model.generate_content(
                        str(messages),
                        stream=True,
                        generation_config=genai.types.GenerationConfig(
                            # candidate_count=1,
                            # stop_sequences=['x'],
                            temperature=temperature,
                            # max_output_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            # seed=seed,
                        )
                    )
                    for chunk in stream:
                        for part in chunk.candidates[0].content.parts:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk.text.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {chunk.text}")

                    # Send final message to close the stream
                    await send({"type": "http.response.body", "body": b'', "more_body": False})

                elif provider == "Groq":
                    stream_kwargs = {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "seed": seed,
                        "stream": True,
                    }

                    stream = await groq_client.chat.completions.create(**stream_kwargs)
                    buffer = []
                    n = 1
                    async for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        buffer.append(token)
                        if len(buffer) == n:
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

                elif provider == "HuggingFace":
                    stream_kwargs = {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "seed": seed,
                        "stream": True,
                    }

                    stream = await hugging_face_client.chat_completion(**stream_kwargs)
                    buffer = []
                    n = 1
                    async for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        buffer.append(token)
                        if len(buffer) == n:
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

                elif provider == "Bedrock":
                    async def generate_request():
                        if model.startswith("cohere"):
                            native_request = {
                                "message": messages[0]["content"],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "p": top_p,
                                "seed": seed,
                            }
                        elif model.startswith("meta"):
                            native_request = {
                                "prompt": messages[0]["content"],
                                "temperature": temperature,
                                "max_gen_len": 2048 if max_tokens > 2048 else max_tokens,
                                "top_p": top_p,
                            }
                        # elif model.startswith("anthropic"):
                        #     native_request = {
                        #         "anthropic_version": "bedrock-2023-05-31",
                        #         "messages": messages,
                        #         "temperature": temperature,
                        #         "max_tokens": max_tokens,
                        #         "top_p": top_p,
                        #     }
                        elif model.startswith("mistral"):
                            native_request = {
                                "prompt": messages[0]["content"],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            }
                        elif model.startswith("amazon"):
                            native_request = {
                                "inputText": messages[0]["content"],
                                "textGenerationConfig": {
                                    "maxTokenCount": max_tokens,
                                    "temperature": temperature,
                                    "topP": top_p,
                                },
                            }
                        elif model.startswith("ai21"):
                            native_request = {
                                "prompt": messages[0]["content"],
                                "maxTokens": max_tokens,
                                "temperature": temperature,
                                "topP": top_p,
                            }
                        request = json.dumps(native_request)
                        return request

                    async def extract_token(chunk):
                        if model.startswith("cohere"):
                            token = chunk.get("text") or ""
                        elif model.startswith("meta"):
                            token = chunk.get("generation") or ""
                        # elif model.startswith("anthropic"):
                        #     # token = chunk.get("completion") or ""
                        #     if chunk['type'] == 'content_block_delta':
                        #         if chunk['delta']['type'] == 'text_delta':
                        #             token = chunk['delta']['text'] or ""
                        elif model.startswith("mistral"):
                            token = chunk.get("outputs")[0]["text"] or ""
                        elif model.startswith("amazon"):
                            token = chunk.get("outputText") or ""
                        elif model.startswith("ai21"):
                            token = chunk.get("completions")[0]["data"]["text"] or ""
                        return token
                    aws_session = aioboto3.Session()
                    aws_bedrock_client = aws_session.client(**bedrock_client_parameters)

                    request = await generate_request()
                    async with aws_bedrock_client as client:
                        stream = await client.invoke_model_with_response_stream(
                            modelId=model, body=request
                        )

                        buffer = []
                        n = 1
                        async for event in stream["body"]:
                            chunk = json.loads(event["chunk"]["bytes"])
                            token = await extract_token(chunk)
                            buffer.append(token)
                            if len(buffer) == n:
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

                else:
                    bt.logging.error(f"Unknown provider: {provider}")

            except Exception as e:
                bt.logging.error(f"error in _prompt {e}\n{traceback.format_exc()}")

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    async def images(self, synapse: ImageResponse) -> ImageResponse:
        bt.logging.info(f"received image request: {synapse}")
        try:
            # Extract necessary information from synapse
            provider = synapse.provider
            model = synapse.model
            messages = synapse.messages
            size = synapse.size
            width = synapse.width
            height = synapse.height
            quality = synapse.quality
            style = synapse.style
            seed = synapse.seed
            steps = synapse.steps
            image_revised_prompt = None
            cfg_scale = synapse.cfg_scale
            sampler = synapse.sampler
            samples = synapse.samples
            image_data = {}

            bt.logging.debug(f"data = {provider, model, messages, size, width, height, quality, style, seed, steps, image_revised_prompt, cfg_scale, sampler, samples}")

            if provider == "OpenAI":
                meta = await client.images.generate(
                    model=model,
                    prompt=messages,
                    size=size,
                    quality=quality,
                    style=style,
                    )
                image_url = meta.data[0].url
                image_revised_prompt = meta.data[0].revised_prompt
                image_data["url"] = image_url
                image_data["image_revised_prompt"] = image_revised_prompt
                bt.logging.info(f"returning image response of {image_url}")

            # elif provider == "Stability":
            #     bt.logging.debug(f"calling stability for {messages, seed, steps, cfg_scale, width, height, samples, sampler}")

            #     meta = stability_api.generate(
            #         prompt=messages,
            #         seed=seed,
            #         steps=steps,
            #         cfg_scale=cfg_scale,
            #         width=width,
            #         height=height,
            #         samples=samples,
            #         # sampler=sampler
            #     )
            #     # Process and upload the image
            #     b64s = []
            #     for image in meta:
            #         for artifact in image.artifacts:
            #             b64s.append(base64.b64encode(artifact.binary).decode())

            #     image_data["b64s"] = b64s
            #     bt.logging.info(f"returning image response to {messages}")

            else:
                bt.logging.error(f"Unknown provider: {provider}")

            synapse.completion = image_data
            return synapse

        except Exception as exc:
            bt.logging.error(f"error in images: {exc}\n{traceback.format_exc()}")

    async def embeddings(self, synapse: Embeddings) -> Embeddings:
        bt.logging.info(f"entered embeddings processing for embeddings of len {len(synapse.texts)}")

        async def get_embeddings_in_batch(texts, model, batch_size=10):
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            tasks = []
            for batch in batches:
                filtered_batch = [text for text in batch if text.strip()]
                if filtered_batch:
                    task = asyncio.create_task(client.embeddings.create(
                        input=filtered_batch, model=model, encoding_format='float'
                    ))
                    tasks.append(task)
                else:
                    bt.logging.info("Skipped an empty batch.")

            all_embeddings = []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(f"Error in processing batch: {result}")
                else:
                    batch_embeddings = [item.embedding for item in result.data]
                    all_embeddings.extend(batch_embeddings)
            return all_embeddings

        try:
            texts = synapse.texts
            model = synapse.model
            batched_embeddings = await get_embeddings_in_batch(texts, model)
            synapse.embeddings = batched_embeddings
            # synapse.embeddings = [np.array(embed) for embed in batched_embeddings]
            bt.logging.info(f"synapse response is {synapse.embeddings[0][:10]}")
            return synapse
        except Exception:
            bt.logging.error(f"Exception in embeddings function: {traceback.format_exc()}")

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.debug("answered to be active")
        synapse.completion = "True"
        return synapse


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


if __name__ == "__main__":
    with StreamMiner():
        while True:
            time.sleep(1)
