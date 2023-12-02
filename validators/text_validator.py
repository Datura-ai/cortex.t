
import math
import torch
import wandb
import random
import asyncio
import template.reward
import bittensor as bt

from base_validator import BaseValidator
from template.protocol import StreamPrompting
from template.utils import call_openai, get_question


class TextValidator(BaseValidator):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet):
        super().__init__(dendrite, metagraph, config, subtensor, wallet, timeout=30)
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

    async def start_query(self, available_uids):
        query_tasks = []
        uid_to_question = {}
        for uid in available_uids:
            prompt = await get_question("text")
            uid_to_question[uid] = prompt
            messages = [{'role': 'user', 'content': prompt}]
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed)
            bt.logging.info(f"Sending {syn.model} {self.query_type} request to uid: {uid}, timeout {self.timeout}: {syn.messages[0]['content']}")
            task = self.query_miner(self.metagraph.axons[uid], uid, syn)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    async def handle_response(self, uid, responses):
        full_response = ""
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, str):  # Now expecting chunk to be a string
                    bt.logging.debug(chunk)
                    full_response += chunk
            bt.logging.info(f"full_response for uid {uid}: {full_response}")
            break
        return uid, full_response

    async def score_responses(self, query_responses, uid_to_question):
        scores = torch.zeros(len(self.metagraph.hotkeys))
        uid_scores_dict = {}
        openai_score_tasks = []

        # Decide to score all UIDs this round based on a chance
        random_number = random.random()
        will_score_all = random_number < 1/1.05
        bt.logging.info(f"Random Number: {random_number}, Will Score All: {will_score_all}")

        for uid, response in query_responses:
            self.wandb_data["responses"][uid] = response
            if will_score_all and response:
                messages = [{'role': 'user', 'content': uid_to_question[uid]}]
                task = call_openai(messages, 0, self.model, self.seed)
                openai_score_tasks.append((uid, task))

        openai_responses = await asyncio.gather(*[task for _, task in openai_score_tasks])

        scoring_tasks = []
        for (uid, _), openai_answer in zip(openai_score_tasks, openai_responses):
            if openai_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = template.reward.openai_score(openai_answer, response, self.weight)
                scoring_tasks.append((uid, task))

        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])

        for (uid, _), score in zip(scoring_tasks, scored_responses):
            scores[uid] = score if score is not None else 0
            uid_scores_dict[uid] = scores[uid]
            self.wandb_data["scores"][uid] = score

        return scores, uid_scores_dict, self.wandb_data

    async def get_and_score(self, available_uids):
        query_responses, uid_to_question = await self.start_query(available_uids)
        return await self.score_responses(query_responses, uid_to_question)