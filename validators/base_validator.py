import asyncio
import bittensor as bt

from abc import ABC, abstractmethod


class BaseValidator(ABC):
    def __init__(self, dendrite, config, subtensor, wallet, timeout):
        self.dendrite = dendrite
        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.timeout = timeout
        self.streaming = False

    async def query_miner(self, axon, uid, syn):
        try:
            responses = await self.dendrite([axon], syn, deserialize=False, timeout=self.timeout, streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    async def handle_response(self, uid, responses):
        return uid, responses

    @abstractmethod
    async def start_query(self, available_uids, metagraph):
        pass

    @abstractmethod
    async def score_responses(self, query_responses, uid_to_question, metagraph):
        pass

    @abstractmethod
    async def get_and_score(self, available_uids, metagraph):
        pass
