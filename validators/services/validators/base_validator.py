from abc import abstractmethod
import asyncio
from datasets import load_dataset
import random
from typing import List, Tuple

import bittensor as bt

from cortext.metaclasses import ValidatorRegistryMeta
from validators.utils import error_handler

dataset = None


class BaseValidator(metaclass=ValidatorRegistryMeta):
    def __init__(self, config, metagraph):
        self.config = config
        self.metagraph = metagraph
        self.dendrite = config.dendrite
        self.wallet = config.wallet
        self.timeout = config.async_time_out
        self.streaming = False
        self.provider = None
        self.model = None
        self.seed = random.randint(1111, 9999)
        self.uid_to_questions = dict()
        self.available_uids = []
        self.num_samples = 100
        self.wandb_data = {}

    async def query_miner(self, metagraph, uid, syn):
        try:
            responses = await self.dendrite([metagraph.axons[uid]], syn, deserialize=False, timeout=self.timeout,
                                            streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    @abstractmethod
    def select_random_provider_and_model(self):
        pass

    async def handle_response(self, uid, response) -> Tuple[int, bt.Synapse]:
        if type(response) == list and response:
            response = response[0]
        return uid, response

    @abstractmethod
    async def create_query(self, uid):
        pass

    @abstractmethod
    async def build_wandb_data(self, scores, responses):
        pass

    def should_i_score(self):
        return True

    @abstractmethod
    async def get_answer_task(self, uid, synapse, response):
        pass

    @abstractmethod
    async def get_scoring_task(self, uid, answer, response):
        pass

    @error_handler
    async def score_responses(self, responses):
        answering_tasks = []
        scoring_tasks = []
        uid_scores_dict = {}
        scored_response = []

        for uid, query_resp in responses:
            task = self.get_answer_task(uid, query_resp.get("query"), query_resp.get("response"))
            answering_tasks.append((uid, task))

        answers_results = await asyncio.gather(*[task for _, task in answering_tasks])

        for (uid, query_resp), answer in zip(responses, answers_results):
            task = self.get_scoring_task(uid, answer, query_resp.get("response"))
            scoring_tasks.append((uid, task))

        # Await all scoring tasks
        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])
        average_score = sum(0 if score is None else score for score in scored_responses) / len(
            scored_responses) if scored_responses else 0
        bt.logging.debug(f"scored responses = {scored_responses}, average score = {average_score}")

        for (uid, _), scored_response in zip(scoring_tasks, scored_responses):
            if scored_response is not None:
                bt.logging.trace(f"scored response is None for uid {uid}")
                uid_scores_dict[uid] = float(scored_response)
            else:
                uid_scores_dict[uid] = 0

        if uid_scores_dict != {}:
            validator_type = self.__class__.__name__
            bt.logging.info(f"{validator_type} scores is {uid_scores_dict}")
        bt.logging.trace("score_responses process completed.")

        return uid_scores_dict, scored_response, responses

    async def get_and_score(self, available_uids: List[int]):
        bt.logging.trace("starting query")
        query_responses = await self.start_query(available_uids)
        bt.logging.trace("scoring query with query responses")
        return await self.score_responses(query_responses)
