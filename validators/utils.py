import random
import time
import aiohttp
import asyncio
import redis
import base64
import hashlib
import inspect
import bittensor as bt

from PIL import Image
from io import BytesIO
from functools import wraps
import traceback

from cortext import ImageResponse, ALL_SYNAPSE_TYPE, REDIS_RESULT_STREAM
from validators.services.cache import QueryResponseCache


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
        except GeneratorExit as err:
            bt.logging.error(f"{err}. {traceback.format_exc()}")
        except Exception as err:
            bt.logging.error(f"{err}. {traceback.format_exc()}")
            return None
        else:
            return result

    @wraps(func)
    def wrapper_sync(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            bt.logging.error(f"{err}. {traceback.format_exc()}")
            return None
        else:
            return result

    if inspect.iscoroutine(func):
        return wrapper
    else:
        return wrapper_sync


async def handle_response_stream(responses) -> tuple[str, str]:
    full_response = ""
    async for chunk in responses:
        if isinstance(chunk, str):
            bt.logging.trace(chunk)
            full_response += chunk
    return full_response


def save_answer_to_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        answer = await func(*args, **kwargs)
        query_syn: ALL_SYNAPSE_TYPE = args[2]
        provider = query_syn.provider
        model = query_syn.model

        cache_service = QueryResponseCache()
        try:
            cache_service.set_cache(question=str(query_syn.json()), answer=str(answer), provider=provider, model=model)
        except Exception as err:
            bt.logging.error(f"Exception during cache for uid {args[1]}, {err}")
        else:
            bt.logging.trace(f"saved answer to cache successfully.")
        finally:
            return answer

    return wrapper


def create_hash_value(input_string):
    # Create a SHA-256 hash object based on random and synpase
    input_string = str(input_string) + str(random.Random().random())
    hash_object = hashlib.sha256()
    # Encode the string to bytes and update the hash object
    hash_object.update(input_string.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_value = hash_object.hexdigest()
    return hash_value


@error_handler
async def get_result_entry_from_redis(redis_client, stream_name, last_id, max_try_cnt):
    result_entries = None
    while max_try_cnt:
        result_entries = redis_client.xread({stream_name: last_id}, block=100)
        await asyncio.sleep(0.1)
        if result_entries:
            break
        else:
            max_try_cnt -= 1
    return result_entries


def find_positive_values(data: dict):
    positive_values = {}

    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            nested_result = find_positive_values(value)
            if nested_result:
                positive_values[key] = nested_result
        elif isinstance(value, (int, float)) and value > 0:
            # Store key-value pairs where the value is greater than 0
            positive_values[key] = value

    return positive_values
