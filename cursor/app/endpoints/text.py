import json
from typing import Any, AsyncGenerator
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from cursor.app.core.config import config
from cursor.app.models import ChatRequest
from cursor.app.core.dendrite import CortexDendrite
from cursor.app.core.query_to_validator import query_miner, query_miner_no_stream
from cursor.app.core.middleware import verify_api_key_rate_limit
import asyncio
import time


async def chat(
        chat_request
) -> StreamingResponse | JSONResponse:
    try:
        if chat_request.stream:
            print(f"stream started. {chat_request}")
            return "hello world"
            return StreamingResponse(query_miner(chat_request), media_type="text/event-stream")
        else:
            resp = await query_miner_no_stream(chat_request)
        return JSONResponse({"choices": [{"message": {"content": resp}}]})
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
