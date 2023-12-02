
import requests
import torch
import wandb
import asyncio
import aiohttp
import datetime
import template.reward
import bittensor as bt

from PIL import Image
from io import BytesIO
from template.utils import get_question
from base_validator import BaseValidator
from template.protocol import ImageResponse


class ImageValidator(BaseValidator):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet):
        super().__init__(dendrite, metagraph, config, subtensor, wallet, timeout=30)
        self.streaming = False
        self.query_type = "images"
        self.model = "dall-e-3"
        self.weight = 1
        self.size = "1792x1024"
        self.quality = "standard"
        self.style = "vivid"

        self.wandb_data = {
            "prompts": {},
            "responses": {},
            "images": {},
            "scores": {},
            "timestamps": {},
        }

    async def start_query(self, available_uids):
        # Query all images concurrently
        query_tasks = []
        uid_to_messages = {}
        for uid in available_uids:
            messages = await get_question("images")
            uid_to_messages[uid] = messages  # Store messages for each UID
            syn = ImageResponse(messages=messages, model=self.model, size=self.size, quality=self.quality, style=self.style)
            bt.logging.info(f"Sending a {self.size} {self.quality} {self.style} {self.query_type} request to uid: {uid} using {syn.model} with timeout {self.timeout}: {syn.messages}")
            task = self.query_miner(self.metagraph.axons[uid], uid, syn)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = messages

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_messages

    async def handle_response(self, uid, responses):
        return uid, responses

    async def download_image(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.read()
                return Image.open(BytesIO(content))

    async def score_responses(self, query_responses, uid_to_messages):
        scores = torch.zeros(len(self.metagraph.hotkeys))
        uid_scores_dict = {}
        download_tasks = []
        score_tasks = []

        for uid, response in query_responses:
            if response:
                response = response[0]
                completion = response.completion
                if completion is not None:
                    bt.logging.info(f"UID {uid} response is {completion}")
                    image_url = completion["url"]

                    # Schedule the image download as an async task
                    download_task = asyncio.create_task(self.download_image(image_url))
                    download_tasks.append((uid, download_task, completion))
                else:
                    bt.logging.info(f"Completion is None for UID {uid}")
                    scores[uid] = 0
                    uid_scores_dict[uid] = 0
            else:
                bt.logging.info(f"No response for UID {uid}")
                scores[uid] = 0
                uid_scores_dict[uid] = 0

        # Wait for all image downloads to complete
        for uid, download_task, completion in download_tasks:
            image = await download_task

            # Log the image to wandb
            self.wandb_data["images"][uid] = wandb.Image(image)
            self.wandb_data["responses"][uid] = completion

            messages_for_uid = uid_to_messages[uid]
            score_task = template.reward.image_score(uid, image_url, self.size, messages_for_uid, self.weight)
            score_tasks.append((uid, asyncio.create_task(score_task)))

        # Wait for all scoring tasks to complete
        scored_responses = await asyncio.gather(*[task for _, task in score_tasks])
        bt.logging.info(f"Scoring tasks completed for UIDs: {[uid for uid, _ in score_tasks]}")

        for (uid, _), score in zip(score_tasks, scored_responses):
            if score is not None:
                scores[uid] = score
                uid_scores_dict[uid] = score
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
            self.wandb_data["scores"][uid] = score
            self.wandb_data["timestamps"][uid] = datetime.datetime.now().isoformat()

        return scores, uid_scores_dict, self.wandb_data

    async def get_and_score(self, available_uids):
        query_responses, uid_to_messages = await self.start_query(available_uids)
        return await self.score_responses(query_responses, uid_to_messages)
