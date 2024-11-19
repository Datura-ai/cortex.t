import json
from typing import Any, AsyncGenerator
import uuid
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis
from fastapi.routing import APIRouter
from cursor.app.models import RequestModel
import asyncio
from redis.asyncio.client import PubSub
import time


def _construct_organic_message(payload: dict, job_id: str, task: str) -> str:
    return json.dumps({"query_type": gcst.ORGANIC, "query_payload": payload, "task": task, "job_id": job_id})


async def _wait_for_acknowledgement(pubsub: PubSub, job_id: str) -> bool:
    async for message in pubsub.listen():
        channel = message["channel"].decode()
        if channel == f"{gcst.ACKNLOWEDGED}:{job_id}" and message["type"] == "message":
            logger.info(f"Job {job_id} confirmed by worker")
            break
    await pubsub.unsubscribe(f"{gcst.ACKNLOWEDGED}:{job_id}")
    return True


async def _stream_results(pubsub: PubSub, job_id: str, task: str, first_chunk: str, start_time: float) -> \
        AsyncGenerator[str, str]:
    yield first_chunk
    num_tokens = 0
    async for message in pubsub.listen():
        channel = message["channel"].decode()

        if channel == f"{rcst.JOB_RESULTS}:{job_id}" and message["type"] == "message":
            result = json.loads(message["data"].decode())
            if gcst.ACKNLOWEDGED in result:
                continue
            status_code = result[gcst.STATUS_CODE]
            if status_code >= 400:
                COUNTER_TEXT_GENERATION_ERROR.add(1, {"task": task, "kind": "nth_chunk_timeout",
                                                      "status_code": status_code})
                raise HTTPException(status_code=status_code, detail=result[gcst.ERROR_MESSAGE])

            content = result[gcst.CONTENT]
            num_tokens += 1
            yield content
            if "[DONE]" in content:
                break
    COUNTER_TEXT_GENERATION_SUCCESS.add(1, {"task": task, "status_code": 200})
    completion_time = time.time() - start_time

    tps = num_tokens / completion_time
    GAUGE_TOKENS_PER_SEC.set(tps, {"task": task})
    logger.info(f"Tokens per second for job_id: {job_id}, task: {task}: {tps}")

    await pubsub.unsubscribe(f"{rcst.JOB_RESULTS}:{job_id}")


async def _get_first_chunk(pubsub: PubSub, job_id: str) -> str | None:
    async for message in pubsub.listen():
        if message["type"] == "message" and message["channel"].decode() == f"{rcst.JOB_RESULTS}:{job_id}":
            result = json.loads(message["data"].decode())
            if gcst.STATUS_CODE in result and result[gcst.STATUS_CODE] >= 400:
                raise HTTPException(status_code=result[gcst.STATUS_CODE], detail=result[gcst.ERROR_MESSAGE])
            return result[gcst.CONTENT]
    return None


async def make_stream_organic_query(
        redis_db: Redis,
        payload: dict[str, Any],
        task: str,
) -> AsyncGenerator[str, str]:
    job_id = uuid.uuid4().hex
    organic_message = _construct_organic_message(payload=payload, job_id=job_id, task=task)

    pubsub = redis_db.pubsub()
    await pubsub.subscribe(f"{gcst.ACKNLOWEDGED}:{job_id}")
    await redis_db.lpush(rcst.QUERY_QUEUE_KEY, organic_message)  # type: ignore

    first_chunk = None
    try:
        await asyncio.wait_for(_wait_for_acknowledgement(pubsub, job_id), timeout=1)
    except asyncio.TimeoutError:
        logger.error(
            f"Query node down? No confirmation received for job {job_id} within timeout period. Task: {task}, model: {payload['model']}"
        )
        COUNTER_TEXT_GENERATION_ERROR.add(1,
                                          {"task": task, "kind": "redis_acknowledgement_timeout", "status_code": 500})
        raise HTTPException(status_code=500, detail="Unable to process request ; redis_acknowledgement_timeout")

    await pubsub.subscribe(f"{rcst.JOB_RESULTS}:{job_id}")
    logger.info("Here waiting for a message!")
    start_time = time.time()
    try:
        first_chunk = await asyncio.wait_for(_get_first_chunk(pubsub, job_id), timeout=2)
    except asyncio.TimeoutError:
        logger.error(
            f"Query node down? Timed out waiting for the first chunk of results for job {job_id}. Task: {task}, model: {payload['model']}"
        )
        COUNTER_TEXT_GENERATION_ERROR.add(1, {"task": task, "kind": "first_chunk_timeout", "status_code": 500})
        raise HTTPException(status_code=500, detail="Unable to process request ; first_chunk_timeout")

    if first_chunk is None:
        COUNTER_TEXT_GENERATION_ERROR.add(1, {"task": task, "kind": "first_chunk_missing", "status_code": 500})
        raise HTTPException(status_code=500, detail="Unable to process request ; first_chunk_missing")
    return _stream_results(pubsub, job_id, task, first_chunk, start_time)


async def _handle_no_stream(text_generator: AsyncGenerator[str, str]) -> JSONResponse:
    all_content = ""
    async for chunk in text_generator:
        chunks = load_sse_jsons(chunk)
        if isinstance(chunks, list):
            for chunk in chunks:
                content = chunk["choices"][0]["delta"]["content"]
                all_content += content
                if content == "":
                    break

    return JSONResponse({"choices": [{"delta": {"content": all_content}}]})


async def chat(
        chat_request: request_models.ChatRequest,
        config: Config = Depends(get_config),
) -> StreamingResponse | JSONResponse:
    payload = request_models.chat_to_payload(chat_request)
    payload.temperature = 0.5

    try:
        text_generator = await make_stream_organic_query(
            redis_db=config.redis_db,
            payload=payload.model_dump(),
            task=payload.model,
        )

        logger.info("Here returning a response!")

        if chat_request.stream:
            return StreamingResponse(text_generator, media_type="text/event-stream")
        else:
            return await _handle_no_stream(text_generator)

    except HTTPException as http_exc:
        COUNTER_TEXT_GENERATION_ERROR.add(1,
                                          {"task": payload.model, "kind": type(http_exc).__name__, "status_code": 500})
        logger.info(f"HTTPException in chat endpoint: {str(http_exc)}")
        raise http_exc

    except Exception as e:
        COUNTER_TEXT_GENERATION_ERROR.add(1, {"task": payload.model, "kind": type(e).__name__, "status_code": 500})
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


router = APIRouter()
router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST", "OPTIONS"],
    tags=["StreamPrompting"],
    response_model=None,
    dependencies=[Depends(verify_api_key_rate_limit)],
)
