import json
from typing import Any, AsyncGenerator
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from app.core.config import config
from app.models import ChatRequest
from app.core.dendrite import CortexDendrite
from cursor.app.core.query_to_validator import query_miner
from cursor.app.core.middleware import verify_api_key_rate_limit
import asyncio
import time

from uaclient.status import status


async def chat(
        chat_request: ChatRequest
) -> StreamingResponse | JSONResponse:
    try:
        return StreamingResponse(await query_miner(chat_request), media_type="text/event-stream")
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail={"message": "internal server error"})


router = APIRouter()
router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST", "OPTIONS"],
    tags=["StreamPrompting"],
    response_model=None
)
