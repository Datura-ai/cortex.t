import asyncio
import random
import traceback
import wandb

import cortext.reward
from cortext.protocol import ImageResponse
from validators.services.validators.base_validator import BaseValidator
from validators import utils
from validators.utils import error_handler
from cortext.utils import get_question
import bittensor as bt


class ImageValidator(BaseValidator):
    def __init__(self, config, metagraph=None):
        super().__init__(config, metagraph)
        self.num_uids_to_pick = 30
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
        self.num_uids_to_pick = 30

        if self.provider == "Stability":
            self.seed = random.randint(1000, 1000000)
            self.model = "stable-diffusion-xl-1024-v1-0"

        elif self.provider == "OpenAI":
            self.model = "dall-e-3"

    def get_provider_to_models(self):
        return "OpenAI", "dall-e-3"

    async def get_question(self):
        question = await get_question("images", 1)
        return question

    async def create_query(self, uid, provider=None, model=None) -> bt.Synapse:
        question = await self.get_question()
        syn = ImageResponse(messages=question, model=model, size=self.size, quality=self.quality,
                            style=self.style, provider=provider, seed=self.seed, steps=self.steps)
        bt.logging.info(f"uid = {uid}, syn = {syn}")
        return syn

    def should_i_score(self):
        rand = random.random()
        return rand < 1 / 1

    async def get_scoring_task(self, uid, answer, response: ImageResponse):
        if response is None:
            bt.logging.trace(f"response is None. so return score with 0 for this uid {uid}.")
            return 0
        if response.provider == "OpenAI":
            completion = response.completion
            if completion is None:
                bt.logging.trace(f"response completion is None for uid {uid}. so return score with 0")
                return 0
            image_url = completion["url"]
            score = await cortext.reward.dalle_score(uid, image_url, self.size, response.messages,
                                                     self.weight)
        else:
            bt.logging.trace(f"not found provider type {response.provider}")
            score = 0  # cortext.reward.deterministic_score(uid, syn, self.weight)
        return score

    async def get_answer_task(self, uid, synapse: ImageResponse, response):
        return synapse

    @error_handler
    async def build_wandb_data(self, scores, responses):
        download_tasks = []
        for uid, syn in responses:
            completion = syn.completion
            if completion is None:
                return self.wandb_data
            if syn.provider == "OpenAI":
                image_url = completion["url"]
                bt.logging.info(f"UID {uid} response = {image_url}")
                download_tasks.append(asyncio.create_task(utils.download_image(image_url)))
            else:  # Stability
                b64s = completion["b64s"]
                bt.logging.info(f"UID {uid} responded with an image")
                for b64 in b64s:
                    download_tasks.append(asyncio.create_task(utils.b64_to_image(b64)))

        download_results = await asyncio.gather(*download_tasks)
        for image, uid in zip(download_results, self.uid_to_questions.keys()):
            self.wandb_data["images"][uid] = wandb.Image(image) if image is not None else ''
            self.wandb_data["prompts"][uid] = self.uid_to_questions[uid]
        return self.wandb_data
