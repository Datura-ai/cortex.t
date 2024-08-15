from abc import abstractmethod
import asyncio
from datasets import load_dataset
import random
from typing import List

import bittensor

from cortext.metaclasses import ValidatorRegistryMeta
from cortext import utils
import cortext
from validators.services.bittensor import bt_validator as bt
from validators.config import app_config
import torch


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

    def get_random_texts(self, dataset_name: str, config_name: str, num_samples: int = 100) -> list[str]:
        dataset = load_dataset(dataset_name, config_name)
        texts = [item['text'] for item in dataset['train']]
        return random.sample(texts, num_samples)

    async def load_questions(self, available_uids, item_type: str = "text", vision=False):
        self.uid_to_questions = dict()

        random_texts = self.get_random_texts('wikitext', 'wikitext-2-v1', 100)
        num_texts_per_uid = len(random_texts) // len(available_uids)

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
                start_index = index * num_texts_per_uid
                end_index = start_index + num_texts_per_uid
                prompt = random_texts[start_index:end_index]
                self.uid_to_questions[uid] = prompt

    @abstractmethod
    def build_synapse(self, question) -> bittensor.synapse:
        pass

    async def query_miner(self, metagraph, uid, syn):
        try:
            responses = await self.dendrite([metagraph.axons[uid]], syn, deserialize=False, timeout=self.timeout,
                                            streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    async def handle_response(self, uid, responses):
        return uid, responses

    async def start_query(self, available_uids: List[int]):
        try:
            uid_to_questions: dict = self.load_questions(available_uids)
            query_tasks = []
            for uid, question in uid_to_questions.items():
                syn = self.build_synapse(question)
                task = self.query_miner(self.metagraph, uid, syn)
                query_tasks.append(task)
                self.update_wandb_data()
            query_responses = await asyncio.gather(*query_tasks)
            return query_responses
        except Exception as err:
            bt.logging.exception(err)

    @abstractmethod
    def build_wandb_data(self, scores, responses):
        pass

    def should_i_score(self):
        return True

    @abstractmethod
    def get_answer_task(self, uid, synapse=None):
        pass

    @abstractmethod
    def get_scoring_task(self, uid, answer, response):
        pass

    async def score_responses(self, responses):
        scores = torch.zeros(len(self.metagraph.hotkeys))
        answering_tasks = []
        scoring_tasks = []
        uid_scores_dict = {}
        will_score_all = self.should_i_score()

        for uid, response in responses:
            task = self.get_answer_task(response)
            answering_tasks.append((uid, task))

        answers_results = await asyncio.gather(*[task for _, task in answering_tasks])

        for (uid, response), answer in zip(responses, answers_results):
            task = self.get_scoring_task(answer, response)
            scoring_tasks.append((uid, task))

        # Await all scoring tasks
        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])
        average_score = sum(scored_responses) / len(scored_responses) if scored_responses else 0
        bt.logging.debug(f"scored responses = {scored_responses}, average score = {average_score}")

        for (uid, _), scored_response in zip(self.uid_to_questions, scored_responses):
            if scored_response is not None:
                scores[uid] = scored_response
                uid_scores_dict[uid] = scored_response
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
        bt.logging.info(f"text_scores is {uid_scores_dict}")

        return scores, uid_scores_dict


async def get_and_score(self, available_uids: List[int], metagraph):
    bt.logging.info("starting query")
    query_responses, uid_to_question = await self.start_query(available_uids, metagraph)
    bt.logging.info("scoring query")
    return await self.score_responses(available_uids, query_responses, uid_to_question, metagraph)
