from __future__ import annotations
import random
import asyncio
import bittensor as bt
import cortext.reward
from cortext import client
from cortext.protocol import Embeddings
from validators.services.validators.base_validator import BaseValidator


class EmbeddingsValidator:
    def __init__(self, config, metagraph=None):
        super().__init__(config, metagraph)
        self.streaming = False
        self.config = config
        self.query_type = "embeddings"
        self.model = "text-embedding-ada-002"
        self.weight = 1

        self.wandb_data = {
            "modality": "embeddings",
            "texts": {},
            "embeddings": {},
            "scores": {},
            "timestamps": {},
        }

    async def call_openai_embeddings(self, model=None, texts='', batch_size=10):

        model = model or self.model

        async def get_embeddings_in_batch(texts, model, batch_size):
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            tasks = []
            for batch in batches:
                filtered_batch = [text for text in batch if text.strip()]
                if filtered_batch:
                    # bt.logging.info("Log prompt.", filtered_batch)
                    task = asyncio.create_task(
                        client.embeddings.create(input=filtered_batch, model=model, encoding_format='float')
                    )
                    tasks.append(task)
                else:
                    bt.logging.info("Skipped an empty batch.")

            all_embeddings = []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(f"Error in processing batch: {result}")
                else:
                    batch_embeddings = [item.embedding for item in result.data]
                    all_embeddings.extend(batch_embeddings)
            return all_embeddings

        all_embeddings = await get_embeddings_in_batch(texts, model, batch_size)
        # for task in asyncio.as_completed(tasks):
        #     try:
        #         response = await task
        #         batch_embeddings = [item.embedding for item in response.data]
        #         all_embeddings.extend(batch_embeddings)
        #     except Exception as e:
        #         bt.logging.error(f"Error in processing batch: {e}")
        return all_embeddings

    async def start_query(self, available_uids) -> tuple[(int, bt.Synapse)] | None:
        if not available_uids:
            return None

        query_tasks = []
        await self.load_questions(available_uids, "embedding")

        for uid, prompt in self.uid_to_questions.items():
            syn = Embeddings(model=self.model, texts=prompt)
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid} "
                f"using {syn.model} with timeout {self.timeout}: {syn.texts[0]}"
            )
            task = self.query_miner(self.metagraph, uid, syn)
            query_tasks.append(task)

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 1.1
        return will_score_all

    async def get_answer_task(self, uid, synapse=None):
        messages = self.uid_to_questions[uid]
        task = await self.call_openai_embeddings(self.model, messages)
        return task

    async def get_scoring_task(self, uid, answer, response):
        task = await cortext.reward.embeddings_score_dot(answer, response.embeddings, self.weight)
        return task

    def init_wandb_data(self):
        self.wandb_data = {
            "modality": "embeddings",
            "texts": {},
            "embeddings": {},
            "scores": {},
            "timestamps": {},
        }

    async def build_wandb_data(self, uid_to_score, responses):
        for uid, score in uid_to_score.items():  # Use scoring_tasks here
            self.wandb_data["scores"][uid] = score
            self.wandb_data["texts"][uid] = self.uid_to_questions[uid]
        return self.wandb_data
