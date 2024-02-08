import asyncio
import concurrent
import itertools
import traceback
import random
from typing import Tuple
import template

import bittensor as bt
import torch
import wandb
import os
import shutil

import argparse
import asyncio
import base64
import copy
import json
import os
import pathlib
import requests
import threading
import time
import traceback
from collections import deque
from functools import partial
from typing import Tuple
import bittensor as bt
import google.generativeai as genai
import wandb
from PIL import Image
from stability_sdk import client
from openai import AsyncOpenAI, OpenAI
from stability_sdk import client as stability_client
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import anthropic
from anthropic_bedrock import AsyncAnthropicBedrock, HUMAN_PROMPT, AI_PROMPT, AnthropicBedrock

import template
from template.protocol import Embeddings, ImageResponse, IsAlive, StreamPrompting, TextPrompting
from template.utils import get_version
import sys

from starlette.types import Send

from template.protocol import IsAlive, StreamPrompting, ImageResponse, Embeddings
from text_validator import TextValidator
from image_validator import ImageValidator
from embeddings_validator import EmbeddingsValidator

iterations_per_set_weights = 5
scoring_organic_timeout = 60


class WeightSetter:
    def __init__(self, loop: asyncio.AbstractEventLoop, dendrite, subtensor, config, wallet, text_vali, image_vali, embed_vali):
        bt.logging.info("starting stream miner")
        self.config = config
        bt.logging.info(f"config:\n{self.config}")
        self.prompt_cache: dict[str, Tuple[str, int]] = {}
        self.request_timestamps = {}
        self.loop = loop
        self.dendrite = dendrite
        self.subtensor = subtensor
        self.wallet = wallet
        self.text_vali = text_vali
        self.image_vali = image_vali
        self.embed_vali = embed_vali
        self.moving_average_scores = None
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.total_scores = torch.zeros(len(self.metagraph.hotkeys))
        self.organic_scoring_tasks = set()
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.loop.create_task(self.consume_organic_scoring())
        # self.loop.create_task(self.perform_synthetic_scoring_and_update_weights())

    def config(self) -> bt.config:
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        return bt.config(parser)

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    def blacklist_prompt( self, synapse: StreamPrompting ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.PROMPT_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_is_alive( self, synapse: IsAlive ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.ISALIVE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_images( self, synapse: ImageResponse ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.IMAGE_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_embeddings( self, synapse: Embeddings ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.EMBEDDING_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    def base_blacklist(self, synapse, blacklist_amt = 1) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            # if hotkey in template.VALIDATOR_API_WHITELIST:
            #     return False,  f"accepting {synapse_type} request from {hotkey}"

            if hotkey == self.wallet.hotkey.ss58_address:
                return False, f"accepting {synapse_type} request from self"

            # # check the stake
            # tao = self.metagraph.neurons[uid].S
            # if tao < blacklist_amt:
            #     return True, f"Blacklisted a low stake {synapse_type} request: {tao} < {blacklist_amt} from {hotkey}"

            # time_window = template.MIN_REQUEST_PERIOD * 60
            # current_time = time.time()

            # if hotkey not in self.request_timestamps:
            #     self.request_timestamps[hotkey] = deque()

            # # Remove timestamps outside the current time window
            # while self.request_timestamps[hotkey] and current_time - self.request_timestamps[hotkey][0] > time_window:
            #     self.request_timestamps[hotkey].popleft()

            # # Check if the number of requests exceeds the limit
            # if len(self.request_timestamps[hotkey]) >= template.MAX_REQUESTS:
            #     return (
            #         True,
            #         f"Request frequency for {hotkey} exceeded: "
            #         f"{len(self.request_timestamps[hotkey])} requests in {template.MIN_REQUEST_PERIOD} minutes. "
            #         f"Limit is {template.MAX_REQUESTS} requests."
            #     )

            # self.request_timestamps[hotkey].append(current_time)

            return True, f"rejecting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")
        synapse.completion = "True"
        return synapse

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

            elif provider == "Stability":
                bt.logging.debug(f"calling stability for {messages, seed, steps, cfg_scale, width, height, samples, sampler}")

                meta = stability_api.generate(
                    prompt=messages,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    samples=samples,
                    # sampler=sampler
                )
                # Process and upload the image
                b64s = []
                for image in meta:
                    for artifact in image.artifacts:
                        b64s.append(base64.b64encode(artifact.binary).decode())

                image_data["b64s"] = b64s
                bt.logging.info(f"returning image response to {messages}")

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

    async def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        bt.logging.info(f"received {synapse}")

        async def _prompt(synapse, send: Send):
            bt.logging.info(
                f"Sending {synapse} request to uid: {synapse.uid}, "
            )    

            async def handle_response(responses):
                full_response = "test response"
                try:
                    for resp in responses:
                        async for chunk in resp:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {chunk}")

                    # Send final message to close the stream
                    await send({"type": "http.response.body", "body": b'', "more_body": False})
                except Exception as e:
                    print(f"Error processing response for uid {e}")
                return full_response

            axon = self.metagraph.axons[synapse.uid]
            responses = self.dendrite.query(
                axons=[axon], 
                synapse=synapse, 
                deserialize=False,
                timeout=synapse.timeout,
                streaming=True,
            )
            return await handle_response(responses)

            response = await query_miner(synapse)
            print(response)
        
        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    def text(self, synapse: TextPrompting) -> TextPrompting:
        synapse.completion =  "completed"
        bt.logging.info("completed")

        synapse = self.dendrite.query(self.metagraph.axons[synapse.uid], synapse, timeout=synapse.timeout)

        bt.logging.info(f"synapse = {synapse}")
        return synapse

    async def consume_organic_scoring(self):
        bt.logging.info("Attaching forward function to axon.")
        self.axon.attach(
            forward_fn=self.prompt,
            blacklist_fn=self.blacklist_prompt
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
        # self.axon.serve(netuid = self.config.netuid, subtensor = self.subtensor)
        self.axon.start()
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
            )
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")
        while True:
            try:
                if self.organic_scoring_tasks:
                    completed, _ = await asyncio.wait(self.organic_scoring_tasks, timeout=1,
                                                      return_when=asyncio.FIRST_COMPLETED)
                    for task in completed:
                        if task.exception():
                            bt.logging.error(
                                f'Encountered in {TextValidator.score_responses.__name__} task:\n'
                                f'{"".join(traceback.format_exception(task.exception()))}'
                            )
                        else:
                            success, data = task.result()
                            if not success:
                                continue
                            self.total_scores += data[0]
                    self.organic_scoring_tasks.difference_update(completed)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                bt.logging.error(f'Encountered in {self.consume_organic_scoring.__name__} loop:\n{traceback.format_exc()}')
                await asyncio.sleep(10)
                

    async def perform_synthetic_scoring_and_update_weights(self):
        while True:
            for steps_passed in itertools.count():
                self.metagraph = await self.run_sync_in_async(lambda: self.subtensor.metagraph(self.config.netuid))

                available_uids = await self.get_available_uids()
                selected_validator = self.select_validator(steps_passed)
                scores, _ = await self.process_modality(selected_validator, available_uids)
                self.total_scores += scores

                steps_since_last_update = steps_passed % iterations_per_set_weights

                if steps_since_last_update == iterations_per_set_weights - 1:
                    await self.update_weights(steps_passed)
                else:
                    bt.logging.info(
                        f"Updating weights in {iterations_per_set_weights - steps_since_last_update - 1} iterations."
                    )

                await asyncio.sleep(100)

    def select_validator(self, steps_passed):
        return self.text_vali if steps_passed % 5 in (0, 1, 2, 3) else self.image_vali

    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item()) for uid in self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())

        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

        return available_uids

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(axon, IsAlive(), deserialize=False, timeout=4)
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID

            bt.logging.trace(f"UID {uid} is not active")
            return None

        except Exception as e:
            bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
            return None

    def shuffled(self, list_: list) -> list:
        list_ = list_.copy()
        random.shuffle(list_)
        return list_

    async def process_modality(self, selected_validator, available_uids):
        uid_list = self.shuffled(list(available_uids.keys()))
        bt.logging.info(f"starting {selected_validator.__class__.__name__} get_and_score for {uid_list}")
        scores, uid_scores_dict, wandb_data = await selected_validator.get_and_score(uid_list, self.metagraph)
        if self.config.wandb_on:
            wandb.log(wandb_data)
            bt.logging.success("wandb_log successful")
        return scores, uid_scores_dict

    async def update_weights(self, steps_passed):
        """ Update weights based on total scores, using min-max normalization for display. """
        bt.logging.info("updated weights")
        avg_scores = self.total_scores / (steps_passed + 1)

        # Normalize avg_scores to a range of 0 to 1
        min_score = torch.min(avg_scores)
        max_score = torch.max(avg_scores)

        if max_score - min_score != 0:
            normalized_scores = (avg_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = torch.zeros_like(avg_scores)

        bt.logging.info(f"normalized_scores = {normalized_scores}")
        # We can't set weights with normalized scores because that disrupts the weighting assigned to each validator class
        # Weights get normalized anyways in weight_utils
        await self.set_weights(avg_scores)

    async def set_weights(self, scores):
        # alpha of .3 means that each new score replaces 30% of the weight of the previous weights
        alpha = .3
        if self.moving_average_scores is None:
            self.moving_average_scores = scores.clone()

        # Update the moving average scores
        self.moving_average_scores = alpha * scores + (1 - alpha) * self.moving_average_scores
        bt.logging.info(f"Updated moving average of weights: {self.moving_average_scores}")
        await self.run_sync_in_async(
            lambda: self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=self.moving_average_scores,
                wait_for_inclusion=False,
                version_key=template.__weights_version__,
            )
        )
        bt.logging.success("Successfully set weights.")
