from abc import ABC, abstractmethod
from typing import List

from validators.services.bittensor import bt_validator as bt
from validators.config import app_config


class BaseValidator(ABC):
    def __init__(self):
        self.dendrite = bt.dendrite
        self.config = bt.config
        self.subtensor = bt.subtensor
        self.wallet = bt.wallet
        self.timeout = app_config.ASYNC_TIME_OUT
        self.streaming = False

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
