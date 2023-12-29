import asyncio
import random
from typing import AsyncIterator, Tuple

import bittensor as bt
import torch
from base_validator import BaseValidator

import template.reward
from template.protocol import StreamPrompting
from template.utils import call_openai, get_question


class TextValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=75)
        self.streaming = True
        self.query_type = "text"
        self.model = "gpt-4-1106-preview"
        self.weight = 1
        self.seed = 1234

        self.wandb_data = {
            "modality": "text",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

    async def organic(self, metagraph, query: dict[str, list[dict[str, str]]]) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed)
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

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}
        for uid in available_uids:
            prompt = await get_question("text", len(available_uids))
            uid_to_question[uid] = prompt
            messages = [{'role': 'user', 'content': prompt}]
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed)
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )
            task = self.query_miner(metagraph.axons[uid], uid, syn)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
    ) -> tuple[torch.Tensor, dict[int, float], dict]:
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        openai_response_tasks = []

        # Decide to score all UIDs this round based on a chance
        random_number = random.random()
        will_score_all = random_number < 1/12
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")

        for uid, response in query_responses:
            self.wandb_data["responses"][uid] = response
            if will_score_all and response:
                prompt = uid_to_question[uid]
                messages = [{'role': 'user', 'content': prompt}]
                task = call_openai(messages, 0, self.model, self.seed)
                openai_response_tasks.append((uid, task))

        openai_responses = await asyncio.gather(*[task for _, task in openai_response_tasks])

        scoring_tasks = []
        for (uid, _), openai_answer in zip(openai_response_tasks, openai_responses):
            if openai_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = template.reward.openai_score(openai_answer, response, self.weight)
                scoring_tasks.append((uid, task))

        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])

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
