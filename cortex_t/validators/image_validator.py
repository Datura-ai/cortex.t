import torch
import wandb
import random
import asyncio
import aiohttp
import base64
import traceback
import cortex_t.template.reward
import bittensor as bt

from PIL import Image
from io import BytesIO

from cortex_t.template import utils
from cortex_t.template.utils import get_question
from cortex_t.validators import base_validator
from cortex_t.validators.base_validator import BaseValidator
from cortex_t.template.protocol import ImageResponse


class MinerOffline(Exception):
    pass


class Provider(base_validator.Provider):
    openai = 'openai'
    stability = 'stability'


provider_to_model = {
    Provider.openai: "dall-e-2",
    Provider.stability: "stable-diffusion-xl-1024-v1-0",
}

provider_to_synapse_provider = {
    Provider.openai: "OpenAI",
    Provider.stability: "Stability",
}


class ImageValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=60)
        self.streaming = False
        self.query_type = "images"
        self.model = "dall-e-2"
        self.weight = .5
        self.provider = "OpenAI"
        self.size = "1024x1024"
        self.width = 1024
        self.height = 1024
        self.quality = "standard"
        self.style = "vivid"
        self.steps = 30
        self.seed = 12345
        self.wandb_data = {
            "modality": "images",
            "prompts": {},
            "responses": {},
            "images": {},
            "scores": {},
            "timestamps": {},
        }

    def stability_seed(self):
        return random.randint(1000, 1000000)

    def cfg_scale(self):
        return 8.0

    def samples(self):
        return 1

    def sampler(self):
        return ""

    async def start_query(self, available_uids, metagraph, provider: Provider):
        try:
            query_tasks = []
            uid_to_question = {}

            if provider in (Provider.openai, Provider.stability):
                model = provider_to_model[provider]
                synapse_provider = provider_to_synapse_provider[provider]
            else:
                raise NotImplementedError(f'Unsupported {provider=}')

            # Query all images concurrently
            for uid in available_uids:
                messages = await get_question("images", len(available_uids))
                uid_to_question[uid] = messages  # Store messages for each UID

                syn = ImageResponse(
                    messages=messages,
                    provider=synapse_provider,
                    model=model,
                    seed=self.stability_seed(),
                    steps=self.steps,
                    cfg_scale=self.cfg_scale(),
                    width=self.width,
                    height=self.height,
                    samples=self.samples(),
                    sampler=self.sampler(),

                    size=self.size,
                    quality=self.quality,
                    style=self.style,
                )
                bt.logging.info(f"uid = {uid}, syn = {syn}")

                # bt.logging.info(
                #     f"Sending a {self.size} {self.quality} {self.style} {self.query_type} request "
                #     f"to uid: {uid} using {syn.model} with timeout {self.timeout}: {syn.messages}"
                # )
                task = self.query_miner(metagraph, uid, syn)
                query_tasks.append(task)
                self.wandb_data["prompts"][uid] = messages

            # Query responses is (uid. syn)
            query_responses = await asyncio.gather(*query_tasks)
            return query_responses, uid_to_question
        except:
            bt.logging.error(f"error in start_query {traceback.format_exc()}")


    async def b64_to_image(self, b64):
        image_data = base64.b64decode(b64)
        return await asyncio.to_thread(Image.open, BytesIO(image_data))

    async def download_image(self, url, session):
        async with session.get(url) as response:
            content = await response.read()
            return await asyncio.to_thread(Image.open, BytesIO(content))

    async def process_download_result(self, uid, download_task):
        try:
            image = await download_task
            self.wandb_data["images"][uid] = wandb.Image(image)
        except Exception as e:
            bt.logging.error(f"Error downloading image for UID {uid}: {e}")

    async def process_score_result(self, uid, score_task, scores, uid_scores_dict):
        try:
            scored_response = await score_task
            score = scored_response if scored_response is not None else 0
            scores[uid] = uid_scores_dict[uid] = score
        except Exception as e:
            bt.logging.error(f"Error scoring image for UID {uid}: {e}")

    async def score_responses(
            self,
            query_responses: list[tuple[int, list[ImageResponse]]],
            uid_to_question: dict[int, str],  # uid -> prompt
            metagraph: bt.metagraph,
            provider: Provider,
    ):
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        download_tasks = []
        score_tasks = []
        rand = random.random()
        will_score_all = rand < 1/1
        bt.logging.info(f"random number = {rand}, will score all = {will_score_all}")
        async with aiohttp.ClientSession() as session:
            for uid, syn in query_responses:
                syn = syn[0]
                completion = syn.completion
                if completion is None:
                    scores[uid] = uid_scores_dict[uid] = 0
                    continue

                if provider == Provider.openai:
                    image_url = completion["url"]
                    bt.logging.info(f"UID {uid} response = {image_url}")
                    download_tasks.append((uid, asyncio.create_task(self.download_image(image_url, session))))
                elif provider == Provider.stability:
                    b64s = completion["b64s"]
                    bt.logging.info(f"UID {uid} responded with an image")
                    for b64 in b64s:
                        download_tasks.append((uid, asyncio.create_task(self.b64_to_image(b64))))
                else:
                    raise NotImplementedError(f'Unsupported {provider=}')

                if will_score_all:
                    if syn.provider == "OpenAI":
                        score_task = cortex_t.template.reward.dalle_score(uid, image_url, self.size, syn.messages,
                                                                          self.weight)
                    else:
                        score_task = cortex_t.template.reward.deterministic_score(uid, syn, self.weight)

                    score_tasks.append((uid, asyncio.create_task(score_task)))

            await asyncio.gather(*(dt[1] for dt in download_tasks), *(st[1] for st in score_tasks))

        download_results = [self.process_download_result(uid, dt) for uid, dt in download_tasks]
        await asyncio.gather(*download_results)

        score_results = [self.process_score_result(uid, st, scores, uid_scores_dict) for uid, st in score_tasks]
        await asyncio.gather(*score_results)

        if uid_scores_dict != {}:
            bt.logging.info(f"scores = {uid_scores_dict}")

        return scores, uid_scores_dict, self.wandb_data

    async def organic(self, metagraph, prompt: str, miner_uid: int, provider: Provider):
        # TODO this is not finished
        if provider in (Provider.openai, Provider.stability):
            model = provider_to_model[provider]
            synapse_provider = provider_to_synapse_provider[provider]
        else:
            raise NotImplementedError(f'Unsupported {provider=}')

        syn = ImageResponse(
            messages=prompt,
            provider=synapse_provider,
            model=model,
            seed=self.stability_seed(),
            steps=self.steps,
            cfg_scale=self.cfg_scale(),
            width=self.width,
            height=self.height,
            samples=self.samples(),
            sampler=self.sampler(),

            size=self.size,
            quality=self.quality,
            style=self.style,
        )


        response = await self.query_miner(metagraph, miner_uid, syn)

        resp_syn = response[1][0]
        completion = resp_syn.completion
        if completion is None:
            raise MinerOffline()
        if provider == Provider.openai:
            image_url = completion["url"]
            download_task = self.download_image(image_url, session)
        elif provider == Provider.stability:
            b64s = completion["b64s"]
            download_task = self.b64_to_image(b64s[0])
        asyncio.create_task()


    async def call_stability(self, prompt: str):
        return await utils.call_stability(
            prompt=prompt,
            seed=self.stability_seed(),
            steps=self.steps,
            cfg_scale=self.cfg_scale(),
            width=self.width,
            height=self.height,
            samples=self.samples(),
            sampler=self.sampler(),
        )