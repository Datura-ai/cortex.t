from abc import abstractmethod
from datasets import load_dataset
import random
from typing import List

from cortext.metaclasses import ValidatorRegistryMeta
from cortext import utils

from validators.services.bittensor import bt_validator as bt
from validators.config import app_config


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
        self.uid_to_question = dict()
        self.available_uids = []

    def get_random_texts(self, dataset_name: str, config_name: str, num_samples: int = 100) -> list[str]:
        dataset = load_dataset(dataset_name, config_name)
        texts = [item['text'] for item in dataset['train']]
        return random.sample(texts, num_samples)

    async def load_questions(self, available_uids, item_type: str = "text", vision=False):
        self.uid_to_question = dict()

        random_texts = self.get_random_texts('wikitext', 'wikitext-2-v1', 100)
        num_texts_per_uid = len(random_texts) // len(available_uids)

        for index, uid in enumerate(available_uids):
            if item_type == "images":
                messages = await utils.get_question("images", len(available_uids))
                content = " ".join(messages)
                self.uid_to_question[uid] = content  # Store messages for each UID
            elif item_type == "text":
                question = await utils.get_question("text", len(available_uids), vision)
                if isinstance(question, str):
                    bt.logging.info(f"Question is str, dict expected: {question}")
                prompt = question.get("prompt")
                image_url = question.get("image")
                self.uid_to_question[uid] = {"prompt": prompt}
                self.uid_to_question[uid]["image"] = image_url
            else:
                start_index = index * num_texts_per_uid
                end_index = start_index + num_texts_per_uid
                prompt = random_texts[start_index:end_index]
                self.uid_to_question[uid] = prompt

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

    @abstractmethod
    async def start_query(self, available_uids: List[int]) -> tuple[list, dict]:
        ...

    @abstractmethod
    async def score_responses(self, responses):
        ...

    async def get_and_score(self, available_uids: List[int], metagraph):
        bt.logging.info("starting query")
        query_responses, uid_to_question = await self.start_query(available_uids, metagraph)
        bt.logging.info("scoring query")
        return await self.score_responses(available_uids, query_responses, uid_to_question, metagraph)
