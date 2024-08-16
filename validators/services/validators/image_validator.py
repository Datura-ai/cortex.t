import asyncio
import random
import traceback

import cortext.reward
from cortext.protocol import ImageResponse
from validators.services.bittensor import bt_validator as bt
from validators.services.validators.base_validator import BaseValidator
from validators import utils


class ImageValidator(BaseValidator):
    def __init__(self):
        super().__init__()
        self.streaming = False
        self.query_type = "images"
        self.model = "dall-e-3"
        self.weight = .5
        self.provider = "OpenAI"
        self.size = "1792x1024"
        self.width = 1024
        self.height = 1024
        self.quality = "standard"
        self.style = "vivid"
        self.steps = 30
        self.seed = 123456
        self.wandb_data = {
            "modality": "images",
            "prompts": {},
            "responses": {},
            "images": {},
            "scores": {},
            "timestamps": {},
        }

    def select_random_provider_and_model(self):
        # Randomly choose the provider based on specified probabilities
        providers = ["OpenAI"] * 100 + ["Stability"] * 0
        self.provider = random.choice(providers)

        if self.provider == "Stability":
            self.seed = random.randint(1000, 1000000)
            self.model = "stable-diffusion-xl-1024-v1-0"

        elif self.provider == "OpenAI":
            self.model = "dall-e-3"

    async def start_query(self, available_uids):
        try:
            query_tasks = []

            self.select_random_provider_and_model()
            await self.load_questions(available_uids, "images")

            # Query all images concurrently
            for uid, content in self.uid_to_questions:
                syn = ImageResponse(messages=content, model=self.model, size=self.size, quality=self.quality,
                                    style=self.style, provider=self.provider, seed=self.seed, steps=self.steps)
                bt.logging.info(f"uid = {uid}, syn = {syn}")
                task = self.query_miner(self.metagraph, uid, syn)
                query_tasks.append(task)

            # Query responses is (uid. syn)
            query_responses = await asyncio.gather(*query_tasks)
            return query_responses, self.uid_to_questions
        except:
            bt.logging.error(f"error in start_query {traceback.format_exc()}")

    def should_i_score(self):
        rand = random.random()
        return rand < 1 / 1

    async def get_scoring_task(self, uid, answer, response: ImageResponse):
        if answer is None:
            return None
        if response.provider == "OpenAI":
            completion = answer.completion
            image_url = completion["url"]
            score_task = cortext.reward.dalle_score(uid, image_url, self.size, response.messages,
                                                    self.weight)
        else:
            score_task = None  # cortext.reward.deterministic_score(uid, syn, self.weight)
        return score_task

    async def get_answer_task(self, uid, synapse=None):
        if not self.should_i_score():
            return None
        return synapse

    async def build_wandb_data(self, resp_synapses):
        download_tasks = []
        for uid, syn in resp_synapses:
            completion = syn.completion
            if syn.provider == "OpenAI":
                image_url = completion["url"]
                bt.logging.info(f"UID {uid} response = {image_url}")
                download_tasks.append(asyncio.create_task(utils.download_image(image_url)))
            else:  # Stability
                b64s = completion["b64s"]
                bt.logging.info(f"UID {uid} responded with an image")
                for b64 in b64s:
                    download_tasks.append(asyncio.create_task(utils.b64_to_image(b64)))

            self.wandb_data["prompts"][uid] = self.uid_to_questions[uid]
