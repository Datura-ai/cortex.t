import asyncio
import random
import traceback
from typing import AsyncIterator, Tuple

import bittensor as bt
import cortext.reward
import torch
from base_validator import BaseValidator
from cortext.protocol import StreamPrompting
from cortext.utils import call_anthropic_bedrock, call_bedrock, call_anthropic, call_gemini, call_groq, call_openai, get_question


class TextValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=75)
        self.streaming = True
        self.query_type = "text"
        self.model = "gpt-4-turbo-2024-04-09"
        self.max_tokens = 4096
        self.temperature = 0.001
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

    async def get_new_question(self, qty, vision=False):
        question = await get_question("text", qty, vision)
        prompt = question.get("prompt")
        image_url = question.get("image")
        return prompt, image_url

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        try:
            query_tasks = []
            uid_to_question = {}
            # Randomly choose the provider based on specified probabilities
            providers = ["OpenAI"] * 100 + ["AnthropicBedrock"] * 0 + ["Gemini"] * 0 + ["Anthropic"] * 0 + ["Groq"] * 0 + ["Bedrock"] * 0
            self.provider = random.choice(providers)

            if self.provider == "AnthropicBedrock":
                # bedrock models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]
                # anthropic models = ["claude-2.1", "claude-2.0", "claude-instant-1.2"]
                # gemini models = ["gemini-pro"]
                self.model = "anthropic.claude-v2:1"
            elif self.provider == "OpenAI":
                self.model = "gpt-4o"
                # self.model = "gpt-4-1106-preview"
                # self.model = "gpt-3.5-turbo"

            elif self.provider == "Gemini":
                self.model = "gemini-pro"

            elif self.provider == "Anthropic":
                self.model = "claude-3-opus-20240229"
                # self.model = "claude-3-sonnet-20240229"
                # self.model = "claude-instant-1.2"

            elif self.provider == "Groq":
                self.model = "gemma-7b-it"
                # self.model = "llama3-70b-8192"
                # self.model = "llama3-8b-8192"
                # self.model = "mixtral-8x7b-32768"

            elif self.provider == "Bedrock":
                self.model = "anthropic.claude-3-sonnet-20240229-v1:0"
                # self.model = "cohere.command-r-v1:0"
                # self.model = "meta.llama2-70b-chat-v1"
                # self.model = "amazon.titan-text-express-v1"
                # self.model = "mistral.mistral-7b-instruct-v0:2"
                # self.model = "ai21.j2-mid-v1"

            bt.logging.info(f"provider = {self.provider}\nmodel = {self.model}")

            vision_models = ["gpt-4o", "claude-3-opus-20240229", "anthropic.claude-3-sonnet-20240229-v1:0"]

            for uid in available_uids:
                messages = [{"role": "user"}]
                if self.model in vision_models:
                    prompt, image_url = await self.get_new_question(len(available_uids), vision=True)
                    uid_to_question[uid] = {
                        "prompt": prompt,
                        "image": image_url,
                    }
                    messages[0]["content"] = prompt
                    messages[0]["image"] = image_url
                else:
                    prompt, _ = await self.get_new_question(len(available_uids))
                    uid_to_question[uid] = {
                        "prompt": prompt,
                    }
                    messages[0]["content"] = prompt

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
                task = self.query_miner(metagraph, uid, syn)
                query_tasks.append(task)
                self.wandb_data["prompts"][uid] = prompt

            query_responses = await asyncio.gather(*query_tasks)
            return query_responses, uid_to_question
        except:
            bt.logging.error(f"error in start_query = {traceback.format_exc()}")

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 12
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")
        return will_score_all

    async def call_api(self, prompt: str, image_url: str | None, provider: str) -> str:
        if provider == "OpenAI":
            return await call_openai(
                [{"role": "user", "content": prompt, "image": image_url}], self.temperature, self.model, self.seed, self.max_tokens
            )
        elif provider == "AnthropicBedrock":
            return await call_anthropic_bedrock(prompt, self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
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
        bt.logging.info("starting wandb logging")
        for uid, response in query_responses:
            self.wandb_data["responses"][uid] = response
            if will_score_all and response:
                question = uid_to_question[uid]
                prompt = question.get("prompt")
                image_url = question.get("image")
                response_tasks.append((uid, self.call_api(prompt, image_url, self.provider)))

        bt.logging.info("finished wandb logging and scoring")
        api_responses = await asyncio.gather(*[task for _, task in response_tasks])
        bt.logging.info("gathered response_tasks for api calls")

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
                self.wandb_data["scores"][uid] = scored_response
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0

        if uid_scores_dict != {}:
            bt.logging.info(f"text_scores is {uid_scores_dict}")
        return scores, uid_scores_dict, self.wandb_data
