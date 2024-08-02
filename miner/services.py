import bittensor as bt
import google.generativeai as genai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from anthropic_bedrock import AsyncAnthropicBedrock
import pathlib

from miner.config import config


class Service:
    def __init__(self):

        self.openai_client = AsyncOpenAI(timeout=config.ASYNC_TIME_OUT, api_key=config.OPENAI_API_KEY)

        self.anthropic_client = AsyncAnthropic(timeout=config.ASYNC_TIME_OUT, api_key=config.ANTHROPIC_API_KEY)

        bedrock_client_parameters = {
            "service_name": 'bedrock-runtime',
            "aws_access_key_id": config.AWS_ACCESS_KEY,
            "aws_secret_access_key": config.AWS_SECRET_KEY,
            "region_name": "us-east-1"
        }

        self.anthropic_bedrock_client = AsyncAnthropicBedrock(timeout=config.ASYNC_TIME_OUT,
                                                              **bedrock_client_parameters)

        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.genai = genai

        self.groq_client = AsyncGroq(timeout=config.ASYNC_TIME_OUT, api_key=config.GROQ_API_KEY)

        # Wandb
        netrc_path = pathlib.Path.home() / ".netrc"
        wandb_api_key = config.WANDB_API_KEY
        bt.logging.info("WANDB_API_KEY is set")
        bt.logging.info("~/.netrc exists:", netrc_path.exists())

        if not wandb_api_key and not netrc_path.exists():
            raise ValueError(
                "Please log in to wandb using `wandb login` or set the WANDB_API_KEY environment variable.")

        self.valid_hotkeys = []
