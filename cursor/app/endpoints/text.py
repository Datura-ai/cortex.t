from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from cursor.app.models import ChatRequest
from cursor.app.core.query_to_validator import query_miner, query_miner_no_stream
import bittensor as bt
import traceback


async def chat(
        chat_request: ChatRequest
) -> StreamingResponse | JSONResponse:
    try:
        if chat_request.stream:
            return StreamingResponse(query_miner(chat_request), media_type="text/event-stream")
        else:
            resp = await query_miner_no_stream(chat_request)
            return JSONResponse({"choices": [{"message": {"content": resp}}]})
    except Exception as err:
        bt.logging.error(f"{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"message": "internal server error"})


router = APIRouter()
router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST", "OPTIONS"],
    tags=["StreamPrompting"],
    response_model=None
)
