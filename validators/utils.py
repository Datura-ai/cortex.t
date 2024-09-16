import time
import aiohttp
import asyncio
import base64
import itertools
import inspect
import bittensor as bt

from PIL import Image
from io import BytesIO
from functools import wraps
import logging

from cortext import ImageResponse


async def download_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            response = await  session.get(url)
            content = await response.read()
            return await asyncio.to_thread(Image.open, BytesIO(content))
    except Exception as e:
        bt.logging.exception(e)


async def b64_to_image(b64):
    image_data = base64.b64decode(b64)
    return await asyncio.to_thread(Image.open, BytesIO(image_data))


def error_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
        except Exception as err:
            logging.exception(err)
            return None

        return result

    return wrapper


async def handle_response_stream(responses) -> tuple[str, str]:
    full_response = ""
    async for chunk in responses:
        if isinstance(chunk, str):
            bt.logging.trace(chunk)
            full_response += chunk
    return full_response


def handle_response(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            response = await func(*args, **kwargs)
            end_time = time.time()
            if inspect.isasyncgen(response):
                return await handle_response_stream(response)
            elif isinstance(response, ImageResponse):
                response.process_time = end_time - start_time
                return response
            else:
                bt.logging.error(f"Not found response type: {type(response)}")
                return None
        except Exception as err:
            bt.logging.exception(f"Exception during query for uid {args[1]}, {err}")
            return None

    return wrapper


def apply_for_time_penalty_to_uid_scores(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        uid_to_scores, scores, resps = await func(*args, **kwargs)
        for uid, query_resp in resps:
            resp_synapse = query_resp.get("response")
            if isinstance(resp_synapse, ImageResponse):
                score = uid_to_scores[uid]
                factor = 64
                max_penalty = 0.5
                if resp_synapse.process_time < 5:
                    bt.logging.trace(f"process time is less than 5 sec. so don't apply penalty for uid {uid}")
                else:
                    penalty = min(max_penalty * pow(resp_synapse.process_time, 1.5) / pow(factor, 1.5), max_penalty)
                    bt.logging.trace(f"penatly {penalty} is applied to miner {uid} "
                                     f"for process time {resp_synapse.process_time}")
                    score -= penalty
                uid_to_scores[uid] = max(score, 0)
        return uid_to_scores, scores, resps

    return wrapper


def get_should_i_score_arr_for_text():
    for i in itertools.count():
        yield (i % 3) == 0


def get_should_i_score_arr_for_image():
    for i in itertools.count():
        yield (i % 1) != 0
