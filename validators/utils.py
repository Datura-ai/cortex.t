import random
import aiohttp
import asyncio
import base64
import hashlib
import inspect
import bittensor as bt

from PIL import Image
from io import BytesIO
from functools import wraps
import traceback

from cortext import ImageResponse, ALL_SYNAPSE_TYPE
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


def save_or_get_answer_from_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        query_syn: ALL_SYNAPSE_TYPE = args[2]
        provider = query_syn.provider
        model = query_syn.model

        cache_service = QueryResponseCache()
        answer = cache_service.get_answer(question=str(query_syn.json()), provider=provider, model=model)
        if answer:
            return answer

        answer = await func(*args, **kwargs)
        try:
            cache_service.set_cache(question=str(query_syn.json()), answer=str(answer), provider=provider, model=model)
        except Exception as err:
            bt.logging.error(f"Exception during cache for uid {args[1]}, {err}")
        else:
            bt.logging.trace(f"saved answer to cache successfully.")
        finally:
            return answer

    return wrapper


def get_query_synapse_from_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        cache_service = QueryResponseCache()
        vali = args[0]
        provider = args[2]
        model = args[3]
        questions_answers = cache_service.get_all_question_to_answers(provider=provider, model=model)
        if not questions_answers or random.random() > 0.1:
            # if no cached question and answers, then create question using openai service with probability 90%
            query_syn = await func(*args, **kwargs)
            return query_syn
        # select one of questions_answers from cache database.
        query, answer = random.choice(questions_answers)
        query_syn = vali.get_synapse_from_json(query)
        return query_syn

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


def update_nested_dict(data, keys, value):
    """
    Updates the value in the nested dictionary or creates the key path if it doesn't exist.

    :param data: The dictionary to update.
    :param keys: A list of keys representing the path in the nested dictionary.
    :param value: The value to set at the specified key path.
    """
    if len(keys) == 1:
        data[keys[0]] = value
    else:
        if keys[0] not in data or not isinstance(data[keys[0]], dict):
            data[keys[0]] = {}
        update_nested_dict(data[keys[0]], keys[1:], value)


def setup_max_capacity(item):
    for key, value in item.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            setup_max_capacity(value)
        elif isinstance(value, (int, float)):  # If the value is a number, increment by 5
            item[key] = min(value, 10)


def get_bandwidth(data, uid, provider, model):
    if data is None:
        return 0
    value = (data.get(uid, {}) or {}).get(provider, {}).get(model, 0)
    return value


def load_entire_questions():
    # Asynchronous function to fetch a URL
    async def fetch(session, url):
        async with session.get(url) as response:
            try:
                return await response.json()
            except Exception as err:
                pass

    # Asynchronous function to gather multiple HTTP requests
    async def gather_requests(urls):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(fetch(session, url))  # Create a task for each URL
            results = await asyncio.gather(*tasks)  # Run all tasks concurrently
            return results

    # Main function to run the event loop
    def main(urls):
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(gather_requests(urls))
        return results

    urls = []
    for q_id in range(0, 80000, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=microsoft%2Fms_marco&config=v1.1&split=train&offset={q_id}&length=100"
        urls.append(url)
    responses = main(urls)
    queries = []
    for response in responses:
        if response is None:
            continue
        for row in response.get('rows', []):
            query = row['row']['query']
            queries.append(query)

    return queries


def create_filtered_message_open_ai(message):
    filtered_message = {
        "role": message["role"],
        "content": [],
    }

    if message.get("content"):
        filtered_message["content"].append(
            {
                "type": "text",
                "text": message["content"],
            }
        )
    if message.get("image"):
        image_url = message.get("image")
        filtered_message["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )

    return filtered_message
