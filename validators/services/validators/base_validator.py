from abc import abstractmethod
import asyncio
from datasets import load_dataset
import random
from typing import List, Tuple

import bittensor

from cortext.metaclasses import ValidatorRegistryMeta
from cortext import utils
from validators.services.bittensor import bt_validator as bt
from validators.config import app_config
import torch

dataset = None


class BaseValidator(metaclass=ValidatorRegistryMeta):
    def __init__(self):
        self.dendrite = bt.dendrite
        self.config = bt.config
        self.subtensor = bt.subtensor
        self.wallet = bt.wallet
        self.metagraph = bt.metagraph
        self.timeout = app_config.ASYNC_TIME_OUT
        self.streaming = False
        self.provider = None
        self.model = None
        self.uid_to_questions = dict()
        self.available_uids = []
        self.num_samples = 100
        self.wandb_data = {}

    def get_random_texts(self) -> list[str]:
        global dataset
        if dataset is None:
            dataset = load_dataset('wikitext', 'wikitext-2-v1')
        texts = [item['text'] for item in dataset['train']]
        return random.sample(texts, self.num_samples)

    async def load_questions(self, available_uids, item_type: str = "text", vision=False):
        self.uid_to_questions = dict()

        for index, uid in enumerate(available_uids):
            if item_type == "images":
                messages = await utils.get_question("images", len(available_uids))
                content = " ".join(messages)
                self.uid_to_questions[uid] = content  # Store messages for each UID
            elif item_type == "text":
                question = await utils.get_question("text", len(available_uids), vision)
                if isinstance(question, str):
                    bt.logging.info(f"Question is str, dict expected: {question}")
                prompt = question.get("prompt")
                image_url = question.get("image")
                self.uid_to_questions[uid] = {"prompt": prompt}
                self.uid_to_questions[uid]["image"] = image_url
            else:
                random_texts = self.get_random_texts()
                num_texts_per_uid = len(random_texts) // len(available_uids)
                start_index = index * num_texts_per_uid
                end_index = start_index + num_texts_per_uid
                prompt = random_texts[start_index:end_index]
                self.uid_to_questions[uid] = prompt

    async def query_miner(self, metagraph, uid, syn):
        try:
            responses = await self.dendrite([metagraph.axons[uid]], syn, deserialize=False, timeout=self.timeout,
                                            streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    async def handle_response(self, uid, response) -> Tuple[int, bittensor.Synapse]:
        if type(response) == list and response:
            response = response[0]
        return uid, response

    @abstractmethod
    async def start_query(self, available_uids: List[int]) -> bittensor.Synapse:
        pass

    @abstractmethod
    async def build_wandb_data(self, scores, responses):
        pass

    def should_i_score(self):
        return True

    @abstractmethod
    async def get_answer_task(self, uid, synapse=None):
        pass

    @abstractmethod
    async def get_scoring_task(self, uid, answer, response):
        pass

    async def score_responses(self, responses):
        answering_tasks = []
        scoring_tasks = []
        uid_scores_dict = {}
        scored_response = []

        for uid, syn in responses:
            task = self.get_answer_task(uid, syn)
            answering_tasks.append((uid, task))

        answers_results = await asyncio.gather(*[task for _, task in answering_tasks])

        for (uid, response), answer in zip(responses, answers_results):
            task = self.get_scoring_task(uid, answer, response)
            scoring_tasks.append((uid, task))

        # Await all scoring tasks
        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])
        average_score = sum(scored_responses) / len(scored_responses) if scored_responses else 0
        bt.logging.debug(f"scored responses = {scored_responses}, average score = {average_score}")

        for (uid, _), scored_response in zip(scoring_tasks, scored_responses):
            if scored_response is not None:
                uid_scores_dict[uid] = scored_response
            else:
                uid_scores_dict[uid] = 0

        if uid_scores_dict != {}:
            bt.logging.info(f"text_scores is {uid_scores_dict}")
        bt.logging.info("score_responses process completed.")

        return uid_scores_dict, scored_response, responses

    async def get_and_score(self, available_uids: List[int]):
        bt.logging.info("starting query")
        query_responses = await self.start_query(available_uids)
        bt.logging.info("scoring query with query responses")
        return await self.score_responses(query_responses)
