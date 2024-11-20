import json
from typing import Any, AsyncGenerator
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from app.core.config import config
from app.models import ChatRequest
from app.core.dendrite import CortexDendrite
import asyncio
import time
import bittensor as bt

subtensor = bt.subtensor(network="finney")
meta = subtensor.metagraph(netuid=18)
print("metagraph synched!")

# This needs to be your validator wallet that is running your subnet 18 validator
# wallet = bt.wallet(name="default", hotkey="default")
wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
dendrite = CortexDendrite(wallet=wallet)


async def chat(
        chat_request: ChatRequest
) -> StreamingResponse | JSONResponse:
    return config


router = APIRouter()
router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST", "OPTIONS"],
    tags=["StreamPrompting"],
    response_model=None
)
