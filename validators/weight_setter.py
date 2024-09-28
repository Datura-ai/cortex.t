import asyncio
import concurrent
import random
import torch
import time

from black.trans import defaultdict
from substrateinterface import SubstrateInterface
from functools import partial
from typing import Tuple, List
import bittensor as bt
from bittensor import StreamingSynapse
import cortext

from starlette.types import Send

from cortext.protocol import IsAlive, StreamPrompting, ImageResponse, Embeddings
from cortext.metaclasses import ValidatorRegistryMeta
from validators.services import CapacityService, BaseValidator, TextValidator, ImageValidator
from validators.services.cache import QueryResponseCache
from validators.utils import error_handler
from validators.task_manager import TaskMgr

scoring_organic_timeout = 60


class WeightSetter:
    def __init__(self, config, cache: QueryResponseCache, loop=None):

        # Cache object using sqlite3.
        self.synthetic_task_done = False
        self.task_mgr: TaskMgr = None
        self.in_cache_processing = False
        self.batch_size = config.max_miners_cnt
        self.cache = cache

        self.uid_to_capacity = {}
        self.available_uid_to_axons = {}
        self.uids_to_query = []
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
        self.max_score_cnt_per_model = 1

        # Initialize scores and counts
        self.total_scores = {}
        self.score_counts = {}
        self.moving_average_scores = None

        # Set up axon and dendrite
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Axon server started on port {self.config.axon.port}")
        self.dendrite = config.dendrite

        # Get network tempo
        self.tempo = self.subtensor.tempo(self.netuid)
        self.weights_rate_limit = self.node_query('SubtensorModule', 'WeightsSetRateLimit', [self.netuid])

        # Set up async-related attributes
        self.lock = asyncio.Lock()
        self.loop = loop or asyncio.get_event_loop()

        # Initialize shared query database
        self.query_database = []

        # initialize uid and capacities.
        asyncio.run(self.initialize_uids_and_capacities())
        # Set up async tasks
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.loop.create_task(self.consume_organic_queries())
        self.loop.create_task(self.perform_synthetic_queries())
        self.loop.create_task(self.process_queries_from_database())

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(None, fn)

    async def refresh_metagraph(self):
        await self.run_sync_in_async(lambda: self.metagraph.sync())

    async def initialize_uids_and_capacities(self):
        self.available_uid_to_axons = await self.get_available_uids()
        self.uids_to_query = list(self.available_uid_to_axons.keys())
        bt.logging.info(f"Available UIDs: {list(self.available_uid_to_axons.keys())}")
        self.uid_to_capacity = await self.get_capacities_for_uids(self.available_uid_to_axons)
        bt.logging.info(f"Capacities for miners: {self.uid_to_capacity}")
        # Initialize total_scores, score_counts.
        self.total_scores = {uid: 0.0 for uid in self.available_uid_to_axons.keys()}
        self.score_counts = {uid: 0 for uid in self.available_uid_to_axons.keys()}

        # update task_mgr after synthetic query at the end of iterator.
        if self.task_mgr:
            self.task_mgr.update_remain_capacity_based_on_new_capacity(self.uid_to_capacity)
        else:
            self.task_mgr = TaskMgr(uid_to_capacities=self.uid_to_capacity, dendrite=self.dendrite,
                                    metagraph=self.metagraph, loop=self.loop)

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value

        except Exception as err:
            # reinitilize node
            self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value

        return result

    def get_blocks_til_epoch(self, block):
        return self.tempo - (block + 19) % (self.tempo + 1)

    def is_epoch_end(self):
        current_block = self.node_query('System', 'Number', [])
        last_update = current_block - self.node_query('SubtensorModule', 'LastUpdate', [self.netuid])[self.my_uid]
        bt.logging.info(f"last update: {last_update} blocks ago")
        if last_update >= self.tempo * 2 or (
                self.get_blocks_til_epoch(current_block) < 10 and last_update >= self.weights_rate_limit):
            return True
        return False

    async def update_and_refresh(self):
        await self.update_weights()
        bt.logging.info("Refreshing metagraph...")
        await self.refresh_metagraph()
        await self.initialize_uids_and_capacities()
        bt.logging.info("Metagraph refreshed.")

    async def query_miner(self, uid, query_syn: cortext.ALL_SYNAPSE_TYPE):
        if query_syn.streaming:
            if uid is None:
                bt.logging.error("Can't create task.")
                return
            bt.logging.trace(f"synthetic task is created and uid is {uid}")

            async def handle_response(responses):
                start_time = time.time()
                response_text = ''
                for resp in responses:
                    async for chunk in resp:
                        if isinstance(chunk, str):
                            response_text += chunk
                            bt.logging.trace(f"Streamed text: {chunk}")

                # Store the query and response in the shared database
                async with self.lock:
                    self.query_database.append({
                        'uid': uid,
                        'synapse': query_syn,
                        'response': (response_text, time.time() - start_time),
                        'query_type': 'organic',
                        'timestamp': asyncio.get_event_loop().time(),
                        'validator': ValidatorRegistryMeta.get_class('TextValidator')(config=self.config,
                                                                                      metagraph=self.metagraph)
                    })

            axon = self.metagraph.axons[uid]
            responses = await self.dendrite(
                axons=[axon],
                synapse=query_syn,
                deserialize=False,
                timeout=query_syn.timeout,
                streaming=True,
            )
            await handle_response(responses)
        else:
            pass

    async def create_query_syns_for_remaining_bandwidth(self):
        query_tasks = []
        for uid, provider_to_cap in self.task_mgr.remain_resources.items():
            for provider, model_to_cap in provider_to_cap.items():
                for model, bandwidth in model_to_cap.items():
                    if bandwidth > 0:
                        # create task and send remaining requests to the miner
                        vali = self.choose_validator_from_model(model)
                        query_task = vali.create_query(uid, provider, model)
                        query_tasks += [query_task] * bandwidth
                    else:
                        continue
        query_synapses = await asyncio.gather(*query_tasks)
        return query_synapses

    async def perform_synthetic_queries(self):
        while True:
            # wait for MIN_REQUEST_PERIOD minutes.
            await asyncio.sleep(cortext.REQUEST_PERIOD * 60)
            current_block = self.node_query('System', 'Number', [])
            bt.logging.info(f"start processing synthetic queries at block {current_block} at time {time.time()}")
            start_time = time.time()
            # don't process any organic query while processing synthetic queries.
            synthetic_tasks = []
            async with self.lock:
                # check available bandwidth and send synthetic requests to all miners.
                query_synapses = await self.create_query_syns_for_remaining_bandwidth()
                for query_syn in query_synapses:
                    uid = self.task_mgr.assign_task(query_syn)
                    synthetic_tasks.append(self.query_miner(uid, query_syn))

            bt.logging.debug(f"{time.time() - start_time} elapsed for creating and submitting synthetic queries.")

            # restore capacities immediately after synthetic query consuming all bandwidth.
            self.task_mgr.restore_capacities_for_all_miners()

            await self.dendrite.aclose_session()
            await asyncio.gather(*synthetic_tasks)

            self.synthetic_task_done = True
            bt.logging.info(
                f"synthetic queries has been processed successfully."
                f"total queries are {len(query_synapses)}")

    def choose_validator_from_model(self, model):
        text_validator = ValidatorRegistryMeta.get_class('TextValidator')(config=self.config, metagraph=self.metagraph)
        # image_validator = ValidatorRegistryMeta.get_class('ImageValidator')(config=self.config,
        #                                                                     metagraph=self.metagraph)
        if model != 'dall-e-3':
            text_validator.model = model
            return text_validator
        # else:
        #     return image_validator

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
        start_time = time.time()
        synapse_response: ImageResponse = await self.dendrite(axon, synapse, deserialize=False,
                                                              timeout=synapse.timeout)
        synapse_response.process_time = time.time() - start_time

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

        return synapse_response

    async def prompt(self, synapse: StreamPrompting) -> StreamingSynapse.BTStreamingResponse:
        bt.logging.info(f"Received {synapse}")

        # Return the streaming response
        async def _prompt(query_synapse: StreamPrompting, send: Send):
            bt.logging.info(f"Sending {synapse} request to uid: {synapse.uid}")

            synapse.deserialize_flag = False
            synapse.streaming = True
            uid = self.task_mgr.assign_task(query_synapse)
            if uid is None:
                bt.logging.error("Can't create task.")
                await send({"type": "http.response.body", "body": b'', "more_body": False})
                return
            bt.logging.trace(f"task is created and uid is {uid}")

            async def handle_response(responses):
                start_time = time.time()
                response_text = ''
                for resp in responses:
                    async for chunk in resp:
                        if isinstance(chunk, str):
                            await send({
                                "type": "http.response.body",
                                "body": chunk.encode("utf-8"),
                                "more_body": True,
                            })
                            response_text += chunk
                            bt.logging.trace(f"Streamed text: {chunk}")

                # Store the query and response in the shared database
                async with self.lock:
                    self.query_database.append({
                        'uid': synapse.uid,
                        'synapse': synapse,
                        'response': (response_text, time.time() - start_time),
                        'query_type': 'organic',
                        'timestamp': asyncio.get_event_loop().time(),
                        'validator': ValidatorRegistryMeta.get_class('TextValidator')(config=self.config,
                                                                                      metagraph=self.metagraph)
                    })

                await send({"type": "http.response.body", "body": b'', "more_body": False})

            axon = self.metagraph.axons[uid]
            await self.dendrite.aclose_session()
            responses = await self.dendrite(
                axons=[axon],
                synapse=synapse,
                deserialize=False,
                timeout=synapse.timeout,
                streaming=True,
            )
            return await handle_response(responses)

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

    def get_scoring_tasks_from_query_responses(self, queries_to_process):

        grouped_query_resps = defaultdict(list)
        validator_to_query_resps = defaultdict(list)
        type_to_validator = {}

        # Process queries outside the lock to prevent blocking
        for query_data in queries_to_process:
            uid = query_data['uid']
            synapse = query_data['synapse']
            response = query_data['response']
            validator = query_data['validator']
            vali_type = type(validator).__name__
            type_to_validator[vali_type] = validator

            provider = synapse.provider
            model = synapse.model

            grouped_key = f"{vali_type}:{uid}:{provider}:{model}"
            grouped_query_resps[grouped_key].append(
                (uid, {'query': synapse, 'response': response}))

        for key, uid_to_query_resps in grouped_query_resps.items():
            vali_type = str(key).split(":")[0]
            if not uid_to_query_resps:
                continue
            query_resp_to_score_for_uids = random.choices(uid_to_query_resps, k=self.max_score_cnt_per_model)
            validator_to_query_resps[vali_type] += query_resp_to_score_for_uids

        score_tasks = []
        for vali_type in type_to_validator:
            validator = type_to_validator[vali_type]
            text_score_task = validator.score_responses(validator_to_query_resps[vali_type], self.uid_to_capacity)
            score_tasks.append(text_score_task)
        return score_tasks

    async def process_queries_from_database(self):
        while True:
            await asyncio.sleep(1)  # Adjust the sleep time as needed
            # accumulate all query results for MIN_REQUEST_PERIOD
            if not self.synthetic_task_done or not self.is_epoch_end():
                bt.logging.trace("no data in query_database. so continue...")
                continue

            bt.logging.info(f"start scoring process...")

            async with self.lock:
                queries_to_process = self.query_database.copy()
                self.query_database.clear()

            self.synthetic_task_done = False

            # with all query_respones, select one per uid, provider, model randomly and score them.
            score_tasks = self.get_scoring_tasks_from_query_responses(queries_to_process)

            resps = await asyncio.gather(*score_tasks)
            resps = [item for item in resps if item is not None]
            # Update total_scores and score_counts
            async with self.lock:
                for uid_scores_dict, _, _ in resps:
                    for uid, score in uid_scores_dict.items():
                        self.total_scores[uid] += score
                        self.score_counts[uid] += 1
            bt.logging.info(f"current total score are {self.total_scores}")
            await self.update_and_refresh()

    @property
    def batch_list_of_all_uids(self):
        uids = list(self.available_uid_to_axons.keys())
        batch_size = self.batch_size
        batched_list = []
        for i in range(0, len(uids), batch_size):
            batched_list.append(uids[i:i + batch_size])
        return batched_list

    @error_handler
    async def process_queries_from_cache_database(self):
        # await self.initialize_uids_and_capacities()
        tasks = []
        for uid in self.uid_to_capacity.keys():
            for provider, model_to_bandwidth in self.uid_to_capacity.get(uid).items():
                for model, bandwidth in model_to_bandwidth.items():
                    vali = self.choose_validator_from_model(model)
                    questions_answers: List[Tuple[str, str]] = self.cache.get_all_question_to_answers(provider, model)
                    if not questions_answers:
                        continue
                    # select one of questions_answers
                    query, answer = random.choice(questions_answers)
                    query_syn = vali.get_synapse_from_json(query)
                    tasks.append(self.score_miners_based_cached_answer(vali, query_syn, answer))

        # process tasks in batch_size to not exceed max request per 2min.
        batched_task_list = []
        for i in range(0, len(tasks), cortext.MAX_REQUESTS):
            batched_task_list.append(tasks[i:i + cortext.MAX_REQUESTS])
        for batch_tasks in batched_task_list:
            start_time = time.time()
            await asyncio.gather(*batch_tasks)
            passed_time = time.time() - start_time
            sleep_time = max(cortext.REQUEST_PERIOD * 60 - passed_time, 1)
            bt.logging.debug(f"wait time {sleep_time} to not exceed max_request {cortext.MAX_REQUESTS} in 2min")
            await  asyncio.sleep(sleep_time)

        bt.logging.info("Successfully complete scoring for all miners with cached data and "
                        f"total score is {self.total_scores}")

    async def score_miners_based_cached_answer(self, vali, query, answer):
        total_query_resps = []
        provider = query.provider
        model = query.model

        async def mock_create_query(*args, **kwargs):
            return query

        origin_ref_create_query = vali.create_query
        origin_ref_provider_to_models = vali.get_provider_to_models

        for batch_uids in self.batch_list_of_all_uids:
            vali.create_query = mock_create_query
            vali.get_provider_to_models = lambda: [(provider, model)]
            query_responses = await self.perform_queries(vali, batch_uids)
            total_query_resps += query_responses

        vali.create_query = origin_ref_create_query
        vali.get_provider_to_models = origin_ref_provider_to_models

        bt.logging.debug(f"total cached query_resps: {len(total_query_resps)}")
        queries_to_process = []
        for uid, response_data in total_query_resps:
            # Decide whether to score this query
            queries_to_process.append({
                'uid': uid,
                'synapse': response_data['query'],
                'response': response_data['response'],
                'query_type': 'synthetic',
                'timestamp': asyncio.get_event_loop().time(),
                'validator': vali
            })

        async def mock_answer(*args, **kwargs):
            return answer

        origin_ref_answer_task = vali.get_answer_task
        vali.get_answer_task = mock_answer
        score_tasks = self.get_scoring_tasks_from_query_responses(queries_to_process)
        responses = await asyncio.gather(*score_tasks)
        vali.get_answer_task = origin_ref_answer_task

        responses = [item for item in responses if item is not None]
        for uid_scores_dict, _, _ in responses:
            for uid, score in uid_scores_dict.items():
                self.total_scores[uid] += score
