import json
from typing import Any, AsyncGenerator
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from cursor.app.core.config import config
from cursor.app.models import ChatRequest
import asyncio
import time


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
    response_model=None,
    dependencies=[Depends(verify_api_key_rate_limit)],
)
