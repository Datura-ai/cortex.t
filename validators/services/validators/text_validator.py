import asyncio
import random
import traceback
from typing import AsyncIterator

from validators.services.bittensor import bt_validator as bt
import constants
import cortext.reward
import torch
from validators.services.validators.base_validator import BaseValidator
from typing import Optional
from cortext.protocol import StreamPrompting
from cortext.utils import (call_anthropic_bedrock, call_bedrock, call_anthropic, call_gemini,
                           call_groq, call_openai, get_question)


class TextValidator(BaseValidator):
    def __init__(self, provider: str = None, model: str = None):
        super().__init__()
        self.streaming = True
        self.query_type = "text"
        self.model = model or constants.TEXT_MODEL
        self.max_tokens = constants.TEXT_MAX_TOKENS
        self.temperature = constants.TEXT_TEMPERATURE
        self.weight = constants.TEXT_WEIGHT
        self.seed = constants.TEXT_SEED
        self.top_p = constants.TEXT_TOP_P
        self.top_k = constants.TEXT_TOP_K
        self.provider = provider or constants.TEXT_PROVIDER
        self.num_uids_to_pick = constants.DEFAULT_NUM_UID_PICK

        self.wandb_data = {
            "modality": "text",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

    async def organic(self, metagraph, query: dict[str, list[dict[str, str]]]) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            syn = StreamPrompting(
                messages=messages,
                model=self.model,
                seed=self.seed,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                provider=self.provider,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )

            # self.wandb_data["prompts"][uid] = messages
            responses = await self.dendrite(
                metagraph.axons[uid],
                syn,
                deserialize=False,
                timeout=self.timeout,
                streaming=self.streaming,
            )

            async for resp in responses:
                if not isinstance(resp, str):
                    continue

                bt.logging.trace(resp)
                yield uid, resp

    async def handle_response(self, uid: str, responses) -> tuple[str, str]:
        full_response = ""
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, str):
                    bt.logging.trace(chunk)
                    full_response += chunk
            bt.logging.debug(f"full_response for uid {uid}: {full_response}")
            break
        return uid, full_response

    async def get_new_question(self, qty, vision):
        question = await get_question("text", qty, vision)
        if isinstance(question, str):
            bt.logging.info(f"Question is str, dict expected: {question}")
        prompt = question.get("prompt")
        image_url = question.get("image")
        return prompt, image_url

    async def start_query(self, available_uids):
        try:
            self.num_uids_to_pick = self.select_random_provider_and_model()
            is_vision_model = self.model in constants.VISION_MODELS
            await self.load_questions(available_uids, "text", is_vision_model)

            uids_to_query = available_uids
            query_tasks = []

            # Randomly choose the provider based on specified probabilities
            self.num_uids_to_pick = self.select_random_provider_and_model()
            bt.logging.info(f"provider = {self.provider}\nmodel = {self.model}")

            if self.num_uids_to_pick < len(available_uids):
                uids_to_query = random.sample(available_uids, self.num_uids_to_pick)

            bt.logging.debug(f"querying {self.num_uids_to_pick} uids: {uids_to_query}")
            for uid in uids_to_query:
                syn, prompt = self.build_synapse(uid)
                task = self.query_miner(self.metagraph, uid, syn)
                query_tasks.append(task)
                self.wandb_data["prompts"][uid] = prompt

            query_responses = await asyncio.gather(*query_tasks)

            return query_responses
        except Exception as err:
            bt.logging.error(f"error in start_query = {err}")

    async def build_synapse(self, uid: int):
        message = {"role": "user"}
        prompt = self.uid_to_questions[uid].get("prompt")
        image_url = self.uid_to_questions[uid].get("image")
        if image_url:
            message["image"] = image_url

        message["content"] = prompt

        syn = StreamPrompting(
            messages=[message],
            model=self.model,
            seed=self.seed,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider=self.provider,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        image_info = f" Image: {syn.messages[0]['image']}" if image_url else ""
        bt.logging.info(
            f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
            f"timeout {self.timeout}. Prompt: {syn.messages[0]['content']}.{image_info}"
        )
        return syn

    def select_random_provider_and_model(self):
        providers = ["OpenAI"] * 45 + ["AnthropicBedrock"] * 0 + ["Gemini"] * 2 + ["Anthropic"] * 18 + [
            "Groq"] * 20 + ["Bedrock"] * 15
        self.provider = random.choice(providers)
        self.num_uids_to_pick = constants.DEFAULT_NUM_UID_PICK

        if self.provider == "AnthropicBedrock":
            self.num_uids_to_pick = constants.DEFAULT_NUM_UID_PICK_ANTHROPIC
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
        return self.num_uids_to_pick

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 1
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")
        return will_score_all

    def build_wandb_data(self, scores, responses):
        for uid, response in responses:
            self.wandb_data["responses"][uid] = response
        for uid in self.available_uids:
            prompt = self.uid_to_questions[uid]
            self.wandb_data["prompts"][uid] = prompt
        for (uid, _), scored_response in zip(self.uid_to_questions, scores):
            if scored_response:
                self.wandb_data["scores"][uid] = scored_response

    async def call_api(self, prompt: str, image_url: Optional[str], provider: str) -> str:
        if provider == "OpenAI":
            return await call_openai(
                [{"role": "user", "content": prompt, "image": image_url}], self.temperature, self.model, self.seed,
                self.max_tokens
            )
        elif provider == "AnthropicBedrock":
            return await call_anthropic_bedrock(prompt, self.temperature, self.model, self.max_tokens, self.top_p,
                                                self.top_k)
        elif provider == "Gemini":
            return await call_gemini(prompt, self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
        elif provider == "Anthropic":
            return await call_anthropic(
                [{"role": "user", "content": prompt, "image": image_url}],
                self.temperature,
                self.model,
                self.max_tokens,
                self.top_p,
                self.top_k,
            )
        elif provider == "Groq":
            return await call_groq(
                [{"role": "user", "content": prompt}],
                self.temperature,
                self.model,
                self.max_tokens,
                self.top_p,
                self.seed,
            )
        elif provider == "Bedrock":
            return await call_bedrock(
                [{"role": "user", "content": prompt, "image": image_url}],
                self.temperature,
                self.model,
                self.max_tokens,
                self.top_p,
                self.seed,
            )
        else:
            bt.logging.error(f"provider {provider} not found")

    async def get_answer_task(self, uid):
        if not self.should_i_score():
            return None
        question = self.uid_to_questions[uid]
        prompt = question.get("prompt")
        image_url = question.get("image")
        return await self.call_api(prompt, image_url, self.provider)

    async def get_scoring_task(self, answer, response):
        return await cortext.reward.api_score(answer, response, self.weight, self.temperature, self.provider)
