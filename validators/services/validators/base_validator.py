from abc import ABC, abstractmethod
from typing import List
import random

from validators.services.bittensor import bt_validator as bt
from validators.config import app_config


class BaseValidator(ABC):
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

    def select_random_provider_and_model(self):
        providers = ["OpenAI"] * 45 + ["AnthropicBedrock"] * 0 + ["Gemini"] * 2 + ["Anthropic"] * 18 + [
            "Groq"] * 20 + ["Bedrock"] * 15
        self.provider = random.choice(providers)
        num_uids_to_pick = 30

        if self.provider == "AnthropicBedrock":
            num_uids_to_pick = 1
            self.model = "anthropic.claude-v2:1"

        elif self.provider == "OpenAI":
            models = ["gpt-4o", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125"]
            self.model = random.choice(models)

        elif self.provider == "Gemini":
            models = ["gemini-pro", "gemini-pro-vision", "gemini-pro-vision-latest"]
            self.model = random.choice(models)

        elif self.provider == "Anthropic":
            models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                      "claude-3-haiku-20240307"]
            self.model = random.choice(models)

        elif self.provider == "Groq":
            models = ["gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
            self.model = random.choice(models)

        elif self.provider == "Bedrock":
            models = [
                "anthropic.claude-3-sonnet-20240229-v1:0", "cohere.command-r-v1:0",
                "meta.llama2-70b-chat-v1", "amazon.titan-text-express-v1",
                "mistral.mistral-7b-instruct-v0:2", "ai21.j2-mid-v1", "anthropic.claude-3-5-sonnet-20240620-v1:0"
                                                                      "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0"
            ]
            self.model = random.choice(models)
        else:
            num_uids_to_pick = None
        return num_uids_to_pick
