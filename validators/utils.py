import aiohttp
import asyncio
import base64
import itertools
import bittensor as bt

from PIL import Image
from io import BytesIO
from functools import wraps
import logging


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


def get_should_i_score_arr_for_text():
    for i in itertools.count():
        yield (i % 5) != 0


def get_should_i_score_arr_for_image():
    for i in itertools.count():
        yield (i % 1) != 0
