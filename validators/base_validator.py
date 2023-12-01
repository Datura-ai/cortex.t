from abc import ABC, abstractmethod
import asyncio
import bittensor as bt


class BaseValidator(ABC):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet, model, timeout):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.model = model
        self.timeout = timeout

    async def query_miner(self, axon, uid, syn, syn_type):
        try:
            bt.logging.info(f"Sent {syn_type} request to uid: {uid} using {syn.model} with timeout {self.timeout}")

            # Implement the common part of the query_miner method
            responses = await self.dendrite([axon], syn, deserialize=False, timeout=self.timeout)

            # Handle the response based on the syn_type
            return await self.handle_response(uid, responses, syn_type)
        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    @abstractmethod
    async def handle_response(self, uid, responses, syn_type):
        # This method needs to be implemented in each subclass
        pass

    @abstractmethod
    async def start_query(self, available_uids):
        # Start the query process; specific for each validator type
        pass

    @abstractmethod
    async def score_responses(self, responses):
        # Scoring logic specific to each validator type
        pass

    async def get_and_score(self, available_uids):
        responses = await self.start_query(available_uids)
        return await self.score_responses(responses)