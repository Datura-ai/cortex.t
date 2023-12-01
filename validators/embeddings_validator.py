import wandb
import torch
import random
import asyncio
import bittensor as bt
import template.reward

from . import client
from datasets import load_dataset
from template.protocol import Embeddings
from base_validator import BaseValidator


async def call_openai_embeddings(model, texts, batch_size=10):
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    tasks = []
    for batch in batches:
        filtered_batch = [text for text in batch if text.strip()]
        if filtered_batch:
            print(filtered_batch)
            task = asyncio.create_task(client.embeddings.create(input=filtered_batch, model=model))
            tasks.append(task)
        else:
            bt.logging.debug("Skipped an empty batch.")
    
    all_embeddings = []
    for task in asyncio.as_completed(tasks):
        try:
            response = await task
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            bt.logging.error(f"Error in processing batch: {e}")
    return all_embeddings


class EmbeddingsValidator(BaseValidator):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet):
        super().__init__(dendrite, metagraph, config, subtensor, wallet, timeout=15)
        self.streaming = False
        self.query_type = "embeddings"
        self.model = "text-embedding-ada-002"
        self.weight = 1

        self.wandb_data = {
            "texts": {},
            "embeddings": {},
            "scores": {},
            "timestamps": {},
        }

    def get_random_texts(self, dataset_name, config_name, num_samples=100):
        dataset = load_dataset(dataset_name, config_name)
        texts = [item['text'] for item in dataset['train']] 
        return random.sample(texts, num_samples)

    async def start_query(self, available_uids):
        query_tasks = []
        uid_to_question = {}
        random_texts = self.get_random_texts('wikitext', 'wikitext-2-v1', 100)
        num_texts_per_uid = len(random_texts) // len(available_uids)

        bt.logging.info(f"Each UID will receive {num_texts_per_uid} texts")

        for index, uid in enumerate(available_uids):
            start_index = index * num_texts_per_uid
            end_index = start_index + num_texts_per_uid
            prompt = random_texts[start_index:end_index]
            uid_to_question[uid] = prompt
            syn = Embeddings(model=self.model, texts=prompt)
            bt.logging.info(f"Sending {self.query_type} request to uid: {uid} using {syn.model} with timeout {self.timeout}: {syn.texts[0]}")
            task = self.query_miner(self.metagraph.axons[uid], uid, syn)
            query_tasks.append(task)
            self.wandb_data["texts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    async def handle_response(self, uid, responses):
        return uid, responses

    async def score_responses(self, query_responses, uid_to_question):
        scores = torch.zeros(len(self.metagraph.hotkeys))
        uid_scores_dict = {}
        score_tasks = []

        random_number = random.random()
        will_score_all = random_number < 1/1.1
        bt.logging.info(f"Random Number: {random_number}, Will Score All: {will_score_all}")

        for uid, response in query_responses:
            if will_score_all and response:
                messages = uid_to_question[uid]
                task = call_openai_embeddings(self.model, messages)
                score_tasks.append((uid, task))

        openai_responses = await asyncio.gather(*[task for _, task in score_tasks])

        for (uid, _), openai_answer in zip(score_tasks, openai_responses):
            response = next(res for u, res in query_responses if u == uid)
            response = response[0]
            if response.embeddings is not None:
                response_embeddings = response.embeddings
                task = template.reward.embeddings_score(openai_answer, response_embeddings, self.weight)
                score_tasks.append((uid, task))
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0

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