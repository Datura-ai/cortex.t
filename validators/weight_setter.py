import asyncio
import concurrent
import itertools
import random
import torch
import traceback
from substrateinterface import SubstrateInterface
from functools import partial
from typing import Tuple
import wandb
import bittensor as bt

import cortext
from cortext.protocol import TextPrompting

from starlette.types import Send

from cortext.protocol import IsAlive, StreamPrompting, ImageResponse, Embeddings
from cortext.metaclasses import ValidatorRegistryMeta
from validators.services import BaseValidator, TextValidator, CapacityService

iterations_per_set_weights = 10
scoring_organic_timeout = 60


class WeightSetter:
    def __init__(self, config):
        self.uid_to_capacity = {}
        self.available_uids = []
        bt.logging.info("Initializing WeightSetter")
        self.config = config
        self.wallet = config.wallet
        self.subtensor = bt.subtensor(config=config)
        self.node = SubstrateInterface(url=config.subtensor.chain_endpoint)
        self.netuid = self.config.netuid
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on subnet: {self.netuid} with uid: {self.my_uid}")

        # Initialize scores
        self.moving_average_scores = None
        self.total_scores = torch.zeros(len(self.metagraph.hotkeys))

        # Set up axon and dendrite
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.start()
        bt.logging.info(f"Axon server started on port {self.config.axon.port}")
        self.dendrite = config.dendrite

        # Set up async-related attributes
        self.loop = asyncio.get_event_loop()
        self.request_timestamps = {}
        self.organic_scoring_tasks = set()

        # Initialize prompt cache
        self.prompt_cache = {}

        # Get network tempo
        self.tempo = self.subtensor.tempo(self.netuid)
        self.weights_rate_limit = self.get_weights_rate_limit()

        # Set up async tasks
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.loop.create_task(self.consume_organic_scoring())
        self.loop.create_task(self.perform_synthetic_scoring_and_update_weights())

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    def get_current_block(self):
        return self.node.query("System", "Number", []).value

    def get_weights_rate_limit(self):
        return self.node.query("SubtensorModule", "WeightsSetRateLimit", [self.netuid]).value

    def get_last_update(self, block):
        try:
            last_update_blocks = block - self.node.query("SubtensorModule", "LastUpdate", [self.netuid]).value[
                self.my_uid]
        except Exception as err:
            bt.logging.error(f"Error getting last update: {traceback.format_exc()}")
            bt.logging.exception(err)
            # means that the validator is not registered yet. The validator should break if this is the case anyways
            last_update_blocks = 1000

        bt.logging.info(f"last set weights successfully {last_update_blocks} blocks ago")
        return last_update_blocks

    def get_blocks_til_epoch(self, block):
        return self.tempo - (block + 19) % (self.tempo + 1)

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

    def base_blacklist(self, synapse, blacklist_amt=20000) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            if hotkey == self.wallet.hotkey.ss58_address:
                return False, f"accepting {synapse_type} request from self"

            elif hotkey in cortext.VALIDATOR_API_WHITELIST:
                return False, f"accepting {synapse_type} request from whitelist: {hotkey}"

            return True, f"rejecting {synapse_type} request from {hotkey}"

        except Exception as err:
            bt.logging.exception(err)

    async def images(self, synapse: ImageResponse) -> ImageResponse:
        bt.logging.info(f"received {synapse}")

        synapse = self.dendrite.query(self.metagraph.axons[synapse.uid], synapse, deserialize=False,
                                      timeout=synapse.timeout)

        bt.logging.info(f"new synapse = {synapse}")
        return synapse

    async def embeddings(self, synapse: Embeddings) -> Embeddings:
        bt.logging.info(f"received {synapse}")

        synapse = await self.dendrite(self.metagraph.axons[synapse.uid], synapse, deserialize=False,
                                      timeout=synapse.timeout)

        bt.logging.info(f"new synapse = {synapse}")
        return synapse

    async def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        bt.logging.info(f"received {synapse}")

        async def _prompt(synapse, send: Send):
            bt.logging.info(
                f"Sending {synapse} request to uid: {synapse.uid}, "
            )

            async def handle_response(responses):
                for resp in responses:
                    async for chunk in resp:
                        if isinstance(chunk, str):
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {chunk}")
                    await send({"type": "http.response.body", "body": b'', "more_body": False})

            axon = self.metagraph.axons[synapse.uid]
            responses = self.dendrite.query(
                axons=[axon],
                synapse=synapse,
                deserialize=False,
                timeout=synapse.timeout,
                streaming=True,
            )
            return await handle_response(responses)

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    def text(self, synapse: TextPrompting) -> TextPrompting:
        synapse.completion = "completed"
        bt.logging.info("completed")

        synapse = self.dendrite.query(self.metagraph.axons[synapse.uid], synapse, deserialize=False,
                                      timeout=synapse.timeout)

        bt.logging.info(f"synapse = {synapse}")
        return synapse

    async def consume_organic_scoring(self):
        bt.logging.info("Attaching forward function to axon.")
        self.axon.attach(
            forward_fn=self.prompt,
            blacklist_fn=self.blacklist_prompt,
        ).attach(
            forward_fn=self.images,
            blacklist_fn=self.blacklist_images,
        ).attach(
            forward_fn=self.embeddings,
            blacklist_fn=self.blacklist_embeddings,
        ).attach(
            forward_fn=self.text,
        )
        self.axon.serve(netuid=self.netuid)
        self.axon.start()
        bt.logging.info(f"Running validator on uid: {self.my_uid}")
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
            except Exception as err:
                bt.logging.exception(err)
                await asyncio.sleep(10)

    async def refresh_metagraph(self):
        await self.run_sync_in_async(lambda: self.metagraph.sync())

    async def perform_synthetic_scoring_and_update_weights(self):
        self.available_uids = await self.get_available_uids()
        self.uid_to_capacity = await self.get_capacities_for_uids(self.available_uids)
        bt.logging.info(f"capacities for miners: {self.uid_to_capacity}")

        while True:
            bt.logging.info("start validating process.")
            for steps_passed in itertools.count():

                selected_validator = self.select_validator()

                uid_to_scores = await self.process_modality(selected_validator, self.available_uids)

                if uid_to_scores is None:
                    bt.logging.info("uid_to_scores is None.")
                    continue

                for uid, score in uid_to_scores.items():
                    self.total_scores[uid] += score

                current_block = self.get_current_block()
                last_update = self.get_last_update(current_block)
                if last_update >= self.tempo * 2 or (
                        self.get_blocks_til_epoch(current_block) < 20 and last_update >= self.weights_rate_limit):
                    bt.logging.info("updating weights...")
                    await self.update_weights(steps_passed)
                    bt.logging.info("refreshing metagraph...")
                    await self.refresh_metagraph()
                    bt.logging.info("refreshing available uids...")
                    self.available_uids = await self.get_available_uids()
                    bt.logging.info("refreshing capacities...")
                    self.uid_to_capacity = await self.get_capacities_for_uids(self.available_uids)

                # if we want to slow down the speed of the validator steps
                await asyncio.sleep(self.config.SLEEP_PER_ITERATION)

    def select_validator(self):
        rand = random.random()
        text_validator = ValidatorRegistryMeta.get_class('TextValidator')(config=self.config, metagraph=self.metagraph)
        image_validator = ValidatorRegistryMeta.get_class('ImageValidator')(config=self.config,
                                                                            metagraph=self.metagraph)
        if rand > self.config.IMAGE_VALIDATOR_CHOOSE_PROBABILITY:
            bt.logging.info("text_validator is selected.")
            return text_validator
        else:
            bt.logging.info("image_validator is selected.")
            return image_validator

    async def get_capacities_for_uids(self, uids):
        capacity_service = CapacityService(metagraph=self.metagraph, dendrite=self.dendrite)
        uid_to_capacity = await capacity_service.query_capacity_to_miners(uids)
        return uid_to_capacity

    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        await self.dendrite.aclose_session()
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item()) for uid in
                 self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())

        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

        bt.logging.info(f"available uids: {available_uids.keys()}")
        if self.config.max_miners_cnt < len(available_uids):
            available_uids = random.sample(list(available_uids.keys()), self.config.max_miners_cnt)

        return available_uids

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(axon, IsAlive(), timeout=4)
            if response.completion == 'True':
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID

            bt.logging.error(f"UID {uid} is not active")
            return None

        except Exception as err:
            bt.logging.error(f"Error checking UID {uid}: {err}")
            return None

    @staticmethod
    def shuffled(list_: list) -> list:
        list_ = list_.copy()
        random.shuffle(list_)
        return list_

    async def process_modality(self, selected_validator: BaseValidator, available_uids):
        if not available_uids:
            bt.logging.info("No available uids.")
            return None
        bt.logging.info(f"starting query {selected_validator.__class__.__name__} for miners {available_uids}")
        query_responses = await selected_validator.start_query(available_uids)

        if not selected_validator.should_i_score():
            bt.logging.info("we don't score this time.")
            return None

        bt.logging.info(f"scoring query with query responses for "
                        f"these uids: {available_uids}")
        uid_scores_dict, scored_responses, responses = await selected_validator.score_responses(query_responses)
        wandb_data = await selected_validator.build_wandb_data(uid_scores_dict, responses)
        if self.config.wandb_on and not wandb_data:
            wandb.log(wandb_data)
            bt.logging.success("wandb_log successful")
        return uid_scores_dict

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
                wait_for_inclusion=True,
                version_key=cortext.__weights_version__,
            )
        )
        bt.logging.success("Successfully included weights in block.")
