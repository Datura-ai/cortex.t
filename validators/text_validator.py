import asyncio
import random
import traceback
from typing import AsyncIterator, Tuple

import bittensor as bt
import torch
from base_validator import BaseValidator

import cortext.reward
from cortext.protocol import StreamPrompting
from cortext.utils import call_openai, get_question, call_anthropic, call_gemini, call_claude


class TextValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=60)
        self.streaming = True
        self.query_type = "text"
        self.model =  "gpt-4-1106-preview"
        self.max_tokens = 4096
        self.temperature = 0.0001
        self.weight = 1
        self.seed = 1234
        self.top_p = 0.01
        self.top_k = 1
        self.provider = "OpenAI"

        self.wandb_data = {
            "modality": "text",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

    async def organic(self, metagraph, query: dict[str, list[dict[str, str]]]) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed, max_tokens=self.max_tokens, temperature=self.temperature, provider=self.provider, top_p=self.top_p, top_k=self.top_k)
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

    async def get_question(self, qty):
        return await get_question("text", qty)

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        try:
            query_tasks = []
            uid_to_question = {}
            # Randomly choose the provider based on specified probabilities
            providers = ["OpenAI"] * 88 + ["Anthropic"] * 02 + ["Gemini"] * 0 + ["Claude"] * 10
            self.provider = random.choice(providers)

            if self.provider == "Anthropic":
                # bedrock models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]
                # claude models = ["claude-2.1", "claude-2.0", "claude-instant-1.2"]
                # gemini models = ["gemini-pro"]
                self.model = "anthropic.claude-v2:1"
            elif self.provider == "OpenAI":
                self.model = "gpt-4-1106-preview"
                # self.model = "gpt-3.5-turbo"

            elif self.provider == "Gemini":
                self.model = "gemini-pro"

            elif self.provider == "Claude":
                self.model = "claude-3-opus-20240229"
                # self.model = "claude-3-sonnet-20240229"
            bt.logging.info(f"provider = {self.provider}\nmodel = {self.model}")
            for uid in available_uids:
                prompt = await self.get_question(len(available_uids))
                uid_to_question[uid] = prompt
                messages = [{'role': 'user', 'content': prompt}]
                syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed, max_tokens=self.max_tokens, temperature=self.temperature, provider=self.provider, top_p=self.top_p, top_k=self.top_k)
                bt.logging.info(
                    f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                    f"timeout {self.timeout}: {syn.messages[0]['content']}"
                )
                task = self.query_miner(metagraph, uid, syn)
                query_tasks.append(task)
                self.wandb_data["prompts"][uid] = prompt

            query_responses = await asyncio.gather(*query_tasks)
            return query_responses, uid_to_question
        except:
            bt.logging.error(f"error in start_query = {traceback.format_exc()}")

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 1
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")
        return will_score_all

    async def call_api(self, prompt: str, provider: str) -> str:
        if provider == "OpenAI":
            return await call_openai([{'role': 'user', 'content': prompt}], self.temperature, self.model, self.seed, self.max_tokens)
        elif provider == "Anthropic":
            return await call_anthropic(prompt, self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
        elif provider == "Gemini":
            return await call_gemini(prompt, self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
        elif provider == "Claude":
            return await call_claude([{'role': 'user', 'content': prompt}], self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
        else:
            bt.logging.error(f"provider {provider} not found")

    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
    ) -> tuple[torch.Tensor, dict[int, float], dict]:

        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        response_tasks = []

        # Decide to score all UIDs this round based on a chance
        will_score_all = self.should_i_score()

        for uid, response in query_responses:
            self.wandb_data["responses"][uid] = response
            if will_score_all and response:
                prompt = uid_to_question[uid]
                response_tasks.append((uid, self.call_api(prompt, self.provider)))

        api_responses = await asyncio.gather(*[task for _, task in response_tasks])

        scoring_tasks = []
        for (uid, _), api_answer in zip(response_tasks, api_responses):
            if api_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = cortext.reward.api_score(api_answer, response, self.weight, self.temperature, self.provider)
                scoring_tasks.append((uid, task))

        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])

        bt.logging.debug(f"scored responses = {scored_responses}")
        for (uid, _), scored_response in zip(scoring_tasks, scored_responses):
            if scored_response is not None:
                scores[uid] = scored_response
                uid_scores_dict[uid] = scored_response
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
            # self.wandb_data["scores"][uid] = score

        if uid_scores_dict != {}:
            bt.logging.info(f"text_scores is {uid_scores_dict}")
        return scores, uid_scores_dict, self.wandb_data


