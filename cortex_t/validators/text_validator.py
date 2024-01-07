import asyncio
import random
from typing import AsyncIterator

import bittensor as bt
import torch
from cortex_t.validators import base_validator
from cortex_t.validators.base_validator import BaseValidator

import cortex_t.template.reward
from cortex_t.template.protocol import StreamPrompting
from cortex_t.template.utils import call_openai, get_question, call_anthropic


class Provider(base_validator.Provider):
    openai = 'openai'
    anthropic = 'anthropic'


provider_to_model = {
    Provider.openai: "gpt-3.5-turbo",  # "gpt-4-1106-preview"
    Provider.anthropic: "anthropic.claude-instant-v1",
}

provider_to_synapse_provider = {
    Provider.openai: "OpenAI",
    Provider.anthropic: "Anthropic",
}


class TextValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=75)
        self.streaming = True
        self.query_type = "text"
        self.max_tokens = 2048
        self.temperature = 0.0001
        self.weight = 1
        self.seed = 1234
        self.top_p = 0.01
        self.top_k = 1

        self.wandb_data = {
            "modality": "text",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

    async def organic(
        self,
        metagraph,
        query: dict[int, list[dict[str, str]]],
        provider: Provider,
    ) -> AsyncIterator[tuple[int, str]]:
        if provider in (Provider.openai, Provider.anthropic):
            model = provider_to_model[provider]
            synapse_provider = provider_to_synapse_provider[provider]
        else:
            raise NotImplementedError(f'Unsupported {provider=}')
        for uid, messages in query.items():
            syn = StreamPrompting(
                messages=messages,
                model=model,
                seed=self.seed,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                provider=synapse_provider,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )

            self.wandb_data["prompts"][uid] = messages
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

    async def start_query(self, available_uids, metagraph, provider: Provider) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}

        if provider in (Provider.openai, Provider.anthropic):
            model = provider_to_model[provider]
            synapse_provider = provider_to_synapse_provider[provider]
        else:
            raise NotImplementedError(f'Unsupported {provider=}')

        for uid in available_uids:
            prompt = await self.get_question(len(available_uids))
            uid_to_question[uid] = prompt
            messages = [{'role': 'user', 'content': prompt}]
            syn = StreamPrompting(messages=messages, model=model, seed=self.seed, max_tokens=self.max_tokens,
                                  temperature=self.temperature, provider=synapse_provider,
                                  top_p=self.top_p, top_k=self.top_k)
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )
            task = self.query_miner(metagraph, uid, syn)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 2
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")
        return will_score_all

    async def call_openai(self, prompt: str) -> str:
        return await call_openai([{'role': 'user', 'content': prompt}], self.temperature,
                                 provider_to_model[Provider.openai], self.seed, self.max_tokens)

    async def call_anthropic(self, prompt: str):
        return await call_anthropic(prompt, self.temperature, provider_to_model[Provider.anthropic],
                                    self.max_tokens, self.top_p, self.top_k)

    async def call_api(self, prompt: str, provider: Provider) -> str:
        if provider == Provider.openai:
            return await self.call_openai(prompt)
        elif provider == Provider.anthropic:
            return await self.call_anthropic(prompt)
        else:
            raise NotImplementedError(f'Unsupported {provider=}')

    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
        provider: Provider,
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
                response_tasks.append((uid, self.call_api(prompt, provider)))

        api_responses = await asyncio.gather(*[task for _, task in response_tasks])

        scoring_tasks = []
        for (uid, _), api_answer in zip(response_tasks, api_responses):
            if api_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = cortex_t.template.reward.api_score(api_answer, response, self.weight)
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


class TestTextValidator(TextValidator):
    def __init__(
        self,
        dendrite,
        config,
        subtensor,
        wallet: bt.wallet,
    ):
        super().__init__(dendrite, config, subtensor, wallet)
        self._openai_prompt_to_contents: dict[str, list[str]] = {}
        self._openai_prompts_used: dict[str, int] = {}

        self._anthropic_prompt_to_contents: dict[str, list[str]] = {}
        self._anthropic_prompts_used: dict[str, int] = {}

        self.questions: list[str] = []
        self._questions_retrieved = -1

    def feed_mock_data(self, openai_prompt_to_contents, anthropic_prompt_to_contents, questions):
        self.questions = questions
        self._questions_retrieved = -1

        self._openai_prompt_to_contents = openai_prompt_to_contents
        self._openai_prompts_used = dict.fromkeys(self._openai_prompt_to_contents, -1)

        self._anthropic_prompt_to_contents = anthropic_prompt_to_contents
        self._anthropic_prompt_to_contents = dict.fromkeys(self._anthropic_prompt_to_contents, -1)

    def should_i_score(self):
        return True

    async def call_openai(self, prompt: str) -> str:
        self._openai_prompts_used[prompt] += 1
        used = self._openai_prompts_used[prompt]
        contents = self._openai_prompt_to_contents[prompt]
        return contents[used % len(contents)]

    async def call_anthropic(self, prompt: str) -> str:
        self._anthropic_prompts_used[prompt] += 1
        used = self._anthropic_prompts_used[prompt]
        contents = self._anthropic_prompt_to_contents[prompt]
        return contents[used % len(contents)]

    async def get_question(self, qty):
        self._questions_retrieved += 1
        return self.questions[self._questions_retrieved % len(self.questions)]

    async def query_miner(self, metagraph, uid, syn: StreamPrompting):
        return uid, await self.call_openai(syn.messages[0]['content'])

    async def organic(
        self,
        metagraph,
        query: dict[str, list[dict[str, str]]],
        provider: Provider,
    ) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            for msg in messages:
                if provider == Provider.openai:
                    resp = await self.call_openai(msg['content'])
                elif provider == Provider.anthropic:
                    resp = await self.call_anthropic(msg['content'])
                else:
                    raise NotImplementedError(f'Unsupported {provider=}')
                yield uid, resp
