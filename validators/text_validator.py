
import math
import wandb
import bittensor as bt
import random
import asyncio
from base_validator import BaseValidator
from template.protocol import StreamPrompting
import template.reward
from template.utils import call_openai, get_question


class TextValidator(BaseValidator):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet):
        super().__init__(dendrite, metagraph, config, subtensor, wallet, timeout=24)
        self.query_type = "text"
        self.model = "gpt-4-1106-preview"
        self.weight = 1
        self.seed = 1234

        self.wandb_data = {
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
            task = self.query_miner(self.metagraph.axons[uid], uid, syn, self.query_type)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    async def handle_response(self, uid, responses):
        full_response = ""
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, list):
                    # bt.logging.info(chunk[0])
                    responses += chunk[0]
            break
        return uid, full_response

    async def score_responses(self, query_responses, uid_to_question):
        scores = {}
        uid_scores_dict = {}
        score_tasks = []

        # Decide to score all UIDs this round based on a 1/8 chance
        random_number = random.random()
        will_score_all = random_number < 1/8
        bt.logging.info(f"Random Number: {random_number}, Will Score All: {will_score_all}")

        for uid, response in query_responses:
            if will_score_all and response:
                messages = [{'role': 'user', 'content': uid_to_question[uid]}]
                task = call_openai(messages, 0, self.model, self.seed)
                score_tasks.append((uid, task))

        openai_responses = await asyncio.gather(*[task for _, task in score_tasks])

        for (uid, _), openai_answer in zip(score_tasks, openai_responses):
            if openai_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = template.reward.openai_score(openai_answer, response, self.weight)
                score_tasks.append((uid, task))

        scored_responses = await asyncio.gather(*[task for _, task in score_tasks])

        for (uid, _), score in zip(score_tasks, scored_responses):
            scores[uid] = score if score is not None else 0
            uid_scores_dict[uid] = scores[uid]
            self.wandb_data["scores"][uid] = score

        if self.config.wandb_on:
            wandb.log(self.wandb_data)

        return scores, uid_scores_dict

    async def get_and_score(self, available_uids):
        query_responses, uid_to_question = await self.start_query(available_uids)
        return await self.score_responses(query_responses, uid_to_question)