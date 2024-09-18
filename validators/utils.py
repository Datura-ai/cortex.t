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

from cortext import ImageResponse, ALL_SYNAPSE_TYPE
from validators.services.cache import cache_service


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
            if inspect.isasyncgen(response):
                result = await handle_response_stream(response)
                return result, time.time() - start_time
            elif isinstance(response, ImageResponse):
                response.process_time = time.time() - start_time
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
                # apply penalty for image task.
                score = uid_to_scores[uid]
                factor = 64
                max_penalty = 0.5
                if resp_synapse.process_time < 5:
                    bt.logging.trace(f"process time is less than 5 sec. so don't apply penalty for uid {uid}")
                else:
                    penalty = min(max_penalty * pow(resp_synapse.process_time, 1.5) / pow(factor, 1.5), max_penalty)
                    bt.logging.trace(f"penalty {penalty} is applied to miner {uid} "
                                     f"for process time {resp_synapse.process_time}")
                    score -= penalty
                uid_to_scores[uid] = max(score, 0)
            elif isinstance(resp_synapse, tuple):
                # apply penalty for streaming task.
                resp_str, process_time = resp_synapse
                total_work_done = len(resp_str)
                chars_per_sec = total_work_done / process_time
                bt.logging.debug(f"speed of streaming is {chars_per_sec} chars per second")

                base_speed = 50
                if chars_per_sec >= base_speed:
                    bt.logging.trace(f"don't apply penalty for this uid {uid}")
                else:
                    max_penalty = 0.5
                    penalty = min((base_speed - chars_per_sec) / base_speed, max_penalty)  # max penalty is 0.5
                    new_score = max(uid_to_scores[uid] - penalty, 0)
                    bt.logging.debug(f"penalty is {penalty}, new_score is {new_score} for uid {uid}")
                    uid_to_scores[uid] = new_score
            else:
                pass

        return uid_to_scores, scores, resps

    return wrapper


def save_answer_to_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        answer = await func(*args, **kwargs)
        query_syn: ALL_SYNAPSE_TYPE = args[2]
        provider  = query_syn.provider
        model = query_syn.model
        try:
            cache_service.set_cache(question=str(query_syn.json()), answer=str(answer), provider=provider, model=model)
        except Exception as err:
            bt.logging.error(f"Exception during cache for uid {args[1]}, {err}")
        else:
            bt.logging.trace(f"saved answer to cache successfully.")
        finally:
            return answer
    return wrapper


def get_should_i_score_arr_for_image():
    for i in itertools.count():
        yield (i % 1) != 0
