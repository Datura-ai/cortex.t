import torch
import wandb
import random
import asyncio
import base64
import traceback
import librosa
import cortext.reward
import bittensor as bt

from io import BytesIO
from cortext.utils import get_question
from base_validator import BaseValidator
from cortext.protocol import TTSResponse
from pydantic import BaseModel

class TTSProvider(BaseModel):
    name: str
    weight: float
    models: tuple[str]
    voices: tuple[str]

providers = (
    TTSProvider(
        name="ElevenLabs",
        weight=1.0,
        models=("eleven_multilingual_v2",),
        voices=("Rachel",)
    ),
)

class TTSValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=30)
        self.asr_model = cortext.reward.get_whisper_model("large-v3", "cpu", "int8", 4096)
        self.sr: int = self.asr_model.feature_extractor.sampling_rate
        self.provider: TTSProvider = None
        self.model: str = None
        self.voice: str = None
        self.wandb_data: dict = {
            "modality": "audio",
            "prompts": {},
            "responses": {},
            "audio": {},
            "scores": {},
            "timestamps": {},
        }

    async def start_query(self, available_uids, metagraph):
        try:
            query_tasks = []
            uid_to_question = {}

            # Randomly choose the provider based on specified probabilities
            self.provider = random.choices(providers, weights=[provider.weight for provider in providers], k=1)[0]
            self.model = self.provider.models[0]
            self.voice = random.choice(self.provider.voices)

            # Query all prompts concurrently

            for uid in available_uids:
                messages = await get_question("tts", len(available_uids))
                uid_to_question[uid] = messages  # Store messages for each UID

                syn = TTSResponse(
                    text=messages,
                    provider=self.provider.name,
                    model=self.model,
                    voice=self.voice,
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
            bt.logging.error(f"error in start_query:\n{traceback.format_exc()}")


    async def score_responses(self, query_responses, uid_to_question, metagraph):
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        rand = random.random()
        will_score_all = rand < 1/1

        for uid, syn in query_responses:
            try:
                syn = syn[0]
                audio_b64 = syn.audio_b64
                if audio_b64 is None:
                    scores[uid] = uid_scores_dict[uid] = 0
                    continue

                bt.logging.info(f"UID {uid} responded with a file")
                with BytesIO(base64.b64decode(audio_b64)) as f:
                    audio, _ = librosa.load(f, sr=self.sr)
                    audio = audio.clip(-1, 1)

                    if will_score_all:
                        wer = cortext.reward.calculate_wer(audio, self.asr_model, uid_to_question[uid])
                        if wer > 0.5:
                            scores[uid] = uid_scores_dict[uid] = 0
                            continue
                        mos = cortext.reward.dnsmos_score(audio, self.sr)
                        scores[uid] = uid_scores_dict[uid] = mos
                        # calculate_odds(sum(scores) / len(scores), len(scores), 3.28, 0.15)

                    self.wandb_data["audio"][uid] = wandb.Audio(audio, self.sr, caption=uid_to_question[uid])

            except:
                bt.logging.debug(f"error in score_responses for uid {uid}:\n{traceback.format_exc()}")

        bt.logging.info(f"Final scores: {uid_scores_dict}")
        bt.logging.info("score_responses process completed.")
        return scores, uid_scores_dict, self.wandb_data

