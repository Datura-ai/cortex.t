
import requests
import wandb
import datetime
import bittensor as bt
import asyncio
from PIL import Image
from io import BytesIO
from base_validator import BaseValidator
from template.protocol import ImageResponse
import template.reward
from template.utils import get_question

class ImageValidator(BaseValidator):
    def __init__(self, dendrite, metagraph, config, subtensor, wallet):
        super().__init__(dendrite, metagraph, config, subtensor, wallet, timeout=30)
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
            task = self.query_miner(self.metagraph.axons[uid], uid, syn, self.query_type)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = messages

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_messages

    async def handle_response(self, uid, responses):
        return uid, responses

    async def score_responses(self, query_responses, uid_to_messages):
        scores = {}
        uid_scores_dict = {}
        score_tasks = []

        for uid, response in query_responses:
            if response:
                response = response[0]
                completion = response.completion
                if completion is not None:
                    bt.logging.info(f"UID {uid} response is {completion}")
                    image_url = completion["url"]

                    # Download the image and store it as a BytesIO object
                    image_response = requests.get(image_url)
                    image_bytes = BytesIO(image_response.content)
                    image = Image.open(image_bytes)

                    # Log the image to wandb
                    self.wandb_data["images"][uid] = wandb.Image(image)
                    self.wandb_data["responses"][uid] = completion

                    messages_for_uid = uid_to_messages[uid]
                    task = template.reward.image_score(uid, image_url, self.size, messages_for_uid, self.weight)
                    score_tasks.append((uid, task))
                else:
                    bt.logging.info(f"Completion is None for UID {uid}")
                    scores[uid] = 0
                    uid_scores_dict[uid] = 0
            else:
                bt.logging.info(f"No response for UID {uid}")
                scores[uid] = 0
                uid_scores_dict[uid] = 0

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

        if self.config.wandb_on:
            wandb.log(self.wandb_data)

        return scores, uid_scores_dict

    async def get_and_score(self, available_uids):
        query_responses, uid_to_messages = await self.start_query(available_uids)
        return await self.score_responses(query_responses, uid_to_messages)
