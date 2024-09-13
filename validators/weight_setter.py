import asyncio
import concurrent
import random
import torch
import traceback
from substrateinterface import SubstrateInterface
from functools import partial
from typing import Tuple
import bittensor as bt

import cortext

from starlette.types import Send

from cortext.protocol import IsAlive, StreamPrompting, ImageResponse, Embeddings
from cortext.metaclasses import ValidatorRegistryMeta
from validators.services import CapacityService, BaseValidator
from validators.utils import handle_response

scoring_organic_timeout = 60


class WeightSetter:
    def __init__(self, config):
        self.available_uids = None
        bt.logging.info("Initializing WeightSetter")
        self.config = config
        self.wallet = config.wallet
        self.subtensor = bt.subtensor(config=config)
        self.node = SubstrateInterface(url=config.subtensor.chain_endpoint)
        self.netuid = self.config.netuid
        self.metagraph = bt.metagraph(netuid=self.netuid, network=config.subtensor.chain_endpoint)
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on subnet: {self.netuid} with uid: {self.my_uid}")

        # Scoring and querying parameters
        self.MIN_SCORED_QUERIES = 10  # Minimum number of times each UID should be scored per epoch
        self.scoring_percent = 0.1  # Percentage of total queries that will be scored
        self.TOTAL_QUERIES_PER_UID = int(self.MIN_SCORED_QUERIES / self.scoring_percent)
        bt.logging.info(f"Each UID will receive {self.TOTAL_QUERIES_PER_UID} total queries, "
                        f"with {self.MIN_SCORED_QUERIES} of them being scored.")

        # Initialize scores and counts
        self.total_scores = {}
        self.score_counts = {}  # Number of times a UID has been scored
        self.total_queries_sent = {}  # Total queries sent to each UID
        self.moving_average_scores = None

        # Set up axon and dendrite
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Axon server started on port {self.config.axon.port}")
        self.dendrite = config.dendrite

        # Set up async-related attributes
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()

        # Initialize shared query database
        self.query_database = []

        # Get network tempo
        self.tempo = self.subtensor.tempo(self.netuid)
        self.weights_rate_limit = self.get_weights_rate_limit()

        # Set up async tasks
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.loop.create_task(self.consume_organic_queries())
        self.loop.create_task(self.perform_synthetic_queries())
        self.loop.create_task(self.process_queries_from_database())

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
            # Means that the validator is not registered yet.
            last_update_blocks = 1000

        bt.logging.trace(f"Last set weights successfully {last_update_blocks} blocks ago")
        return last_update_blocks

    def get_blocks_til_epoch(self, block):
        return self.tempo - (block + 19) % (self.tempo + 1)

    async def refresh_metagraph(self):
        await self.run_sync_in_async(lambda: self.metagraph.sync())

    async def initialize_uids_and_capacities(self):
        self.available_uids = await self.get_available_uids()
        bt.logging.info(f"Available UIDs: {list(self.available_uids.keys())}")
        self.uid_to_capacity = await self.get_capacities_for_uids(self.available_uids)
        bt.logging.info(f"Capacities for miners: {self.uid_to_capacity}")
        # Initialize total_scores, score_counts, and total_queries_sent
        self.total_scores = {uid: 0.0 for uid in self.available_uids.keys()}
        self.score_counts = {uid: 0 for uid in self.available_uids.keys()}
        self.total_queries_sent = {uid: 0 for uid in self.available_uids.keys()}

    async def update_and_refresh(self, last_update):
        bt.logging.info(f"Setting weights, last update {last_update} blocks ago")
        await self.update_weights()

        bt.logging.info("Refreshing metagraph...")
        await self.refresh_metagraph()

        bt.logging.info("Refreshing available UIDs...")
        new_available_uids = await self.get_available_uids()
        bt.logging.info(f"Available UIDs: {list(new_available_uids.keys())}")

        bt.logging.info("Refreshing capacities...")
        self.uid_to_capacity = await self.get_capacities_for_uids(new_available_uids)

        # Update total_scores, score_counts, and total_queries_sent
        # Remove UIDs that are no longer available
        for uid in list(self.total_scores.keys()):
            if uid not in new_available_uids:
                del self.total_scores[uid]
                del self.score_counts[uid]
                del self.total_queries_sent[uid]

        # Add new UIDs
        for uid in new_available_uids:
            if uid not in self.total_scores:
                self.total_scores[uid] = 0.0
                self.score_counts[uid] = 0
                self.total_queries_sent[uid] = 0

        # Reset counts for new epoch
        for uid in self.total_scores.keys():
            self.total_scores[uid] = 0.0
            self.score_counts[uid] = 0
            self.total_queries_sent[uid] = 0

        self.available_uids = new_available_uids

    async def perform_synthetic_queries(self):
        while True:
            if self.available_uids is None:
                await self.initialize_uids_and_capacities()

            current_block = self.get_current_block()
            last_update = self.get_last_update(current_block)

            if last_update >= self.tempo * 2 or (
                    self.get_blocks_til_epoch(current_block) < 10 and last_update >= self.weights_rate_limit):
                await self.update_and_refresh(last_update)

            # Decide which UIDs to query, considering total queries sent
            async with self.lock:
                # Select UIDs that have not reached TOTAL_QUERIES_PER_UID
                uids_to_query = [uid for uid in self.available_uids
                                 if self.total_queries_sent[uid] < self.TOTAL_QUERIES_PER_UID]

                if not uids_to_query:
                    bt.logging.info("All UIDs have received the maximum number of total queries.")
                    await asyncio.sleep(1)
                    continue

                # Prioritize UIDs with least total_queries_sent
                uids_to_query.sort(key=lambda uid: self.total_queries_sent[uid])

            # Limit the number of UIDs to query based on configuration
            num_uids_to_query = min(self.config.max_miners_cnt, len(uids_to_query))
            uids_to_query = uids_to_query[:num_uids_to_query]

            selected_validator = self.select_validator()
            selected_validator.select_random_provider_and_model()

            # Perform synthetic queries
            bt.logging.info("start querying to miners")
            query_responses = await self.perform_queries(selected_validator, uids_to_query)

            # Store queries and responses in the shared database
            async with self.lock:
                for uid, response_data in query_responses:
                    # Update total_queries_sent
                    self.total_queries_sent[uid] += 1

                    # Decide whether to score this query
                    if self.should_i_score():
                        self.query_database.append({
                            'uid': uid,
                            'synapse': response_data['query'],
                            'response': response_data['response'],
                            'query_type': 'synthetic',
                            'timestamp': asyncio.get_event_loop().time(),
                            'validator': selected_validator
                        })
                    # If not scoring, we can still log the query if needed

            bt.logging.info(f"Performed synthetic queries for UIDs: {uids_to_query}")

            # Slow down the validator steps if necessary
            await asyncio.sleep(1)

    def should_i_score(self):
        # Randomly decide whether to score this query based on scoring_percent
        return random.random() < self.scoring_percent

    async def perform_queries(self, selected_validator, uids_to_query):
        query_responses = []
        tasks = []
        async def query_miner_per_uid(uid):
            try:
                query_syn = await selected_validator.create_query(uid)
                response = await self.query_miner(uid, query_syn)
                query_responses.append((uid, {'query': query_syn, 'response': response}))
            except Exception as e:
                bt.logging.error(f"Exception during query for uid {uid}: {e}")

        for uid in uids_to_query:
            task = query_miner_per_uid(uid)
            tasks.append(task)
        await asyncio.gather(*tasks)
        return query_responses

    @handle_response
    async def query_miner(self, uid, synapse):
        axon = self.metagraph.axons[uid]

        streaming = False
        if isinstance(synapse, bt.StreamingSynapse):
            streaming = True

        responses = await self.dendrite(
            axons=[axon],
            synapse=synapse,
            deserialize=False,
            timeout=synapse.timeout,
            streaming=streaming,
        )
        # Handle the response appropriately
        return responses[0]  # Assuming responses is a list

    def select_validator(self) -> BaseValidator:
        rand = random.random()
        text_validator = ValidatorRegistryMeta.get_class('TextValidator')(config=self.config, metagraph=self.metagraph)
        image_validator = ValidatorRegistryMeta.get_class('ImageValidator')(config=self.config,
                                                                            metagraph=self.metagraph)
        if rand > self.config.image_validator_probability:
            return text_validator
        else:
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

        bt.logging.info(f"Available UIDs: {list(available_uids.keys())}")

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

    async def update_weights(self):
        """Update weights based on average scores."""
        bt.logging.info("Updating weights...")
        avg_scores = {}

        # Compute average scores per UID
        async with self.lock:
            for uid in self.total_scores:
                count = self.score_counts[uid]
                if count > 0:
                    avg_scores[uid] = self.total_scores[uid] / count
                else:
                    avg_scores[uid] = 0.0

        bt.logging.info(f"Average scores = {avg_scores}")

        # Convert avg_scores to a tensor aligned with metagraph UIDs
        weights = torch.zeros(len(self.metagraph.uids))
        for uid, score in avg_scores.items():
            weights[uid] = score

        await self.set_weights(weights)

    async def set_weights(self, scores):
        # Alpha of .3 means that each new score replaces 30% of the weight of the previous weights
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

    def blacklist_prompt(self, synapse: StreamPrompting) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.PROMPT_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_images(self, synapse: ImageResponse) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.IMAGE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_embeddings(self, synapse: Embeddings) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, cortext.EMBEDDING_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
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
        bt.logging.info(f"Received {synapse}")

        axon = self.metagraph.axons[synapse.uid]
        synapse_response = await self.dendrite(axon, synapse, deserialize=False,
                                               timeout=synapse.timeout)

        bt.logging.info(f"New synapse = {synapse_response}")
        # Store the query and response in the shared database
        async with self.lock:
            self.query_database.append({
                'uid': synapse.uid,
                'synapse': synapse,
                'response': synapse_response,
                'query_type': 'organic',
                'timestamp': asyncio.get_event_loop().time(),
                'validator': ValidatorRegistryMeta.get_class('ImageValidator')(config=self.config,
                                                                               metagraph=self.metagraph)
            })
            # Update total_queries_sent
            self.total_queries_sent[synapse.uid] += 1

        return synapse_response

    async def embeddings(self, synapse: Embeddings) -> Embeddings:
        bt.logging.info(f"Received {synapse}")

        axon = self.metagraph.axons[synapse.uid]
        synapse_response = await self.dendrite(axon, synapse, deserialize=False,
                                               timeout=synapse.timeout)

        bt.logging.info(f"New synapse = {synapse_response}")
        # Store the query and response in the shared database
        async with self.lock:
            self.query_database.append({
                'uid': synapse.uid,
                'synapse': synapse,
                'response': synapse_response,
                'query_type': 'organic',
                'timestamp': asyncio.get_event_loop().time(),
                'validator': ValidatorRegistryMeta.get_class('EmbeddingsValidator')(config=self.config,
                                                                                    metagraph=self.metagraph)
            })
            # Update total_queries_sent
            self.total_queries_sent[synapse.uid] += 1

        return synapse_response

    async def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        bt.logging.info(f"Received {synapse}")

        # Return the streaming response
        async def _prompt(synapse, send: Send):
            bt.logging.info(f"Sending {synapse} request to uid: {synapse.uid}")

            axon = self.metagraph.axons[synapse.uid]
            responses = await self.dendrite(
                axons=[axon],
                synapse=synapse,
                deserialize=False,
                timeout=synapse.timeout,
                streaming=True,
            )

            response_text = ''

            for resp in responses:
                async for chunk in resp:
                    if isinstance(chunk, str):
                        await send({
                            "type": "http.response.body",
                            "body": chunk.encode("utf-8"),
                            "more_body": True,
                        })
                        bt.logging.info(f"Streamed text: {chunk}")
                        response_text += chunk

            await send({"type": "http.response.body", "body": b'', "more_body": False})

            # Store the query and response in the shared database
            async with self.lock:
                self.query_database.append({
                    'uid': synapse.uid,
                    'synapse': synapse,
                    'response': response_text,
                    'query_type': 'organic',
                    'timestamp': asyncio.get_event_loop().time(),
                    'validator': ValidatorRegistryMeta.get_class('TextValidator')(config=self.config,
                                                                                  metagraph=self.metagraph)
                })
                # Update total_queries_sent
                self.total_queries_sent[synapse.uid] += 1

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    async def consume_organic_queries(self):
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
        )
        self.axon.serve(netuid=self.netuid)
        self.axon.start()
        bt.logging.info(f"Running validator on uid: {self.my_uid}")

    async def process_queries_from_database(self):
        while True:
            await asyncio.sleep(1)  # Adjust the sleep time as needed
            async with self.lock:
                if not self.query_database:
                    continue
                # Copy queries to process and clear the database
                queries_to_process = self.query_database.copy()
                self.query_database.clear()

            # Process queries outside the lock to prevent blocking
            for query_data in queries_to_process:
                uid = query_data['uid']
                synapse = query_data['synapse']
                response = query_data['response']
                validator = query_data['validator']

                # Prepare query response data in the format expected by the validator
                query_responses = [(uid, {'query': synapse, 'response': response})]

                # Score the response using the validator
                try:
                    uid_scores_dict, _, _ = await validator.score_responses(query_responses)
                except Exception as e:
                    bt.logging.error(f"Error scoring response for UID {uid}: {e}")
                    continue

                # Update total_scores and score_counts
                async with self.lock:
                    for uid, score in uid_scores_dict.items():
                        self.total_scores[uid] += score
                        self.score_counts[uid] += 1

                        # Stop scoring if MIN_SCORED_QUERIES reached
                        if self.score_counts[uid] >= self.MIN_SCORED_QUERIES:
                            bt.logging.info(f"UID {uid} has reached the minimum number of scored queries.")
