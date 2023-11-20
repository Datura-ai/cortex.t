import os
import re
import json
import math
import time
import torch
import wandb
import random
import string
import asyncio
import template
import requests
import argparse
import datetime
import traceback
import bittensor as bt
import concurrent.futures
from PIL import Image
from io import BytesIO
from openai import AsyncOpenAI
from typing import List, Optional
from template.protocol import StreamPrompting, IsAlive, ImageResponse

AsyncOpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not AsyncOpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=30.0)
list_update_lock = asyncio.Lock()


global state
global config
moving_average_scores = None
state = template.utils.load_state_from_file()

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

def init_wandb(my_subnet_uid):
    if config.wandb_on:
        run_name = f'validator-{my_subnet_uid}'
        config.run_name = run_name
        config.version = template.__version__
        global wandb_run
        wandb_run = wandb.init(
            name=run_name,
            anonymous="allow",
            reinit=False,
            project='synthetic-QA',
            entity='cortex-t',
            config=config,
            dir=config.full_path,
        )
        bt.logging.success('Started wandb run')

def initialize_components(config):
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(config.netuid)
    return wallet, subtensor, dendrite, metagraph

def check_validator_registration(wallet, subtensor, metagraph):
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register --netuid 18 and try again.")
        exit()

async def call_openai(messages, temperature, engine, seed=1234):
    for attempt in range(2):
        bt.logging.debug("Calling Openai")
        try:
            response = await client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            response = response.choices[0].message.content
            bt.logging.trace(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI: {e}")
            await asyncio.sleep(0.5) 
    
    return None

async def get_list(list_type, theme=None):

    list_type_mapping = {
        "text_themes": {
            "default": template.question_themes,
            "prompt": "Create a Python list of 50 unique and thought-provoking themes, each suitable for generating meaningful text-based questions. Limit each theme to a maximum of four words. The themes should be diverse and encompass a range of topics, including technology, philosophy, society, history, science, and art. Format the themes as elements in a Python list, and provide only the list without any additional text or explanations."    
        },
        "images_themes": {
            "default": template.image_themes,
            "prompt": "Generate a Python list of 50 unique and broad creative themes for artistic inspiration. Each theme should be no more than four words, open to interpretation, and suitable for various artistic expressions. Present the list in a single-line Python list structure."
        },
        "text_questions": {
            "default": template.text_questions,
            "prompt": f"Generate a Python list of 20 creative and thought-provoking questions, each related to the theme '{theme}'. Ensure each question is concise, no more than 15 words, and tailored to evoke in-depth exploration or discussion about '{theme}'. Format the output as elements in a Python list, and include only the list without any additional explanations or text."
        },
        "images_questions": {
            "default": template.image_questions,
            "prompt": f"Provide a Python list of 20 creative and detailed scenarios for image generation, each inspired by the theme '{theme}'. The scenarios should be diverse, encompassing elements such as natural landscapes, historical settings, futuristic scenes, and imaginative contexts related to '{theme}'. Each element in the list should be a concise but descriptive scenario, designed to inspire visually rich images. Format these as elements in a Python list."
        }
    }

     # Check if list_type is valid
    if list_type not in list_type_mapping:
        bt.logging.error("no valid list_type provided")
        return
    
    default = list_type_mapping[list_type]["default"]
    prompt = list_type_mapping[list_type]["prompt"]

    messages = [{'role': "user", 'content': prompt}]
    max_retries = 3
    for retry in range(max_retries):
        try:
            random_seed = random.randint(1, 10000)
            answer = await call_openai(messages, .33, "gpt-3.5-turbo", random_seed)
            answer = answer.replace("\n", " ") if answer else ""
            extracted_list = template.utils.extract_python_list(answer)
            if extracted_list:
                bt.logging.info(f"Received {list_type}: {extracted_list}")
                return extracted_list
            else:
                bt.logging.info(f"No valid python list found, retry count: {retry + 1}")
        except Exception as e:
            retry += 1
            bt.logging.error(f"Got exception when calling openai {e}\n{traceback.format_exc()}")

    bt.logging.error(f"No list found after {max_retries} retries, using default list.")
    return default

async def update_counters_and_get_new_list(category, item_type, theme=None):
    global list_update_lock

    async def get_items(category, item_type, theme=None):
        if item_type == "themes":
            return await get_list(f"{category}_themes")
        else:
            # Ensure theme is always available for 'questions'
            if theme is None:
                theme = await get_current_theme(category)
                if theme is None:
                    raise ValueError("No theme available for questions")
            return await get_list(f"{category}_questions", theme)

    async def get_current_theme(category):
        themes = state[category]["themes"]
        if not themes:
            themes = await get_items(category, "themes")
            state[category]["themes"] = themes
        return themes.pop() if themes else None

    list_type = f"{category}_{item_type}"

    async with list_update_lock:
        items = state[category][item_type]

        # Logging the current state before fetching new items
        bt.logging.debug(f"Queue for {list_type}: {len(items) if items else 0} items")

        # Fetch new items if the list is empty
        if not items:
            items = await get_items(category, item_type, theme)
            state[category][item_type] = items
            bt.logging.debug(f"Fetched new list for {list_type}, containing {len(items)} items")

        item = items.pop() if items else None
        if not items:
            state[category][item_type] = None

    return item

async def get_question(category):
    if category not in ["text", "images"]:
        raise ValueError("Invalid category. Must be 'text' or 'images'.")

    question = await update_counters_and_get_new_list(category, "questions")
    return question

async def check_uid(dendrite, axon, uid):
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=2.3)
        if response.is_success:
            bt.logging.debug(f"UID {uid} is active")
            return uid
        else:
            bt.logging.debug(f"UID {uid} is not active")
            return None
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return None

async def get_available_uids(dendrite, metagraph):
    """Get a list of available UIDs asynchronously."""
    tasks = [check_uid(dendrite, metagraph.axons[uid.item()], uid.item()) for uid in metagraph.uids]
    uids = await asyncio.gather(*tasks)
    # Filter out None values (inactive UIDs)
    return [uid for uid in uids if uid is not None]

def set_weights(scores, config, subtensor, wallet, metagraph):
    global moving_average_scores
    alpha = .3
    if moving_average_scores is None:
        moving_average_scores = scores.clone()

    # Update the moving average scores
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores
    bt.logging.info(f"Updated moving average of weights: {moving_average_scores}")
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=moving_average_scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights based on moving average.")

async def query_image(dendrite, axon, uid, syn, config, subtensor, wallet):
    try:
        bt.logging.info(f"Sent image request to uid: {uid}, {syn.messages}")
        responses = await dendrite([axon], syn, deserialize=False, timeout=50)
        return uid, responses  # Return a tuple of the UID and the responses
    except Exception as e:
        bt.logging.error(f"Exception during query for uid {uid}: {e}")
        return uid, None 

async def get_and_score_images(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids):
    engine = "dall-e-3"
    weight = 1
    size = "1024x1024"
    quality = "standard"
    style = "vivid"

    wandb_data = {
        "prompts": {},
        "responses": {},
        "images": {},
        "scores": {},
        "timestamps": {},
    }

    # Step 1: Query all images concurrently
    query_tasks = []
    uid_to_messages = {}
    for uid in available_uids:
        messages = await get_question("images")
        # if wanting a specific image, redefine messages here with an image prompt and comment out the above line
        # messages = "aquamarine and pink, religious symbolism, american works on paper 1880–1950, flowerpunk, colorful assemblages"
        uid_to_messages[uid] = messages  # Store messages for each UID
        syn = ImageResponse(messages=messages, engine=engine, size=size, quality=quality, style=style)
        task = query_image(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)
        query_tasks.append(task)
        wandb_data["prompts"][uid] = messages


    query_responses = await asyncio.gather(*query_tasks)

    # Step 2: Create scoring tasks for all images
    score_tasks = []
    for uid, response in query_responses:
        if response:
            response = response[0]
            completion = response.completion
            if completion is not None:
                bt.logging.info(f"UID {uid} response is {completion}")
                image_url = completion["url"]

                # Download the image and store it as a BytesIO object
                image_response = requests.get(image_url)
                image_bytes = BytesIO(image_response.content)
                image = Image.open(image_bytes)

                # Log the image to wandb
                wandb_data["images"][uid] = wandb.Image(image)
                wandb_data["responses"][uid] = completion

                messages_for_uid = uid_to_messages[uid]
                task = template.reward.image_score(uid, image_url, size, messages_for_uid, weight)
                score_tasks.append((uid, task))
            else:
                bt.logging.info(f"Completion is None for UID {uid}")
                scores[uid] = 0
                uid_scores_dict[uid] = 0
        else:
            bt.logging.info(f"No response for UID {uid}")
            scores[uid] = 0
            uid_scores_dict[uid] = 0

    # Step 3: Run all scoring tasks concurrently
    scored_responses = await asyncio.gather(*[task for _, task in score_tasks])
    bt.logging.info(f"Scoring tasks completed for UIDs: {[uid for uid, _ in score_tasks]}")

    # Step 4: Update scores and uid_scores_dict
    for (uid, _), score in zip(score_tasks, scored_responses):
        if score is not None:
            scores[uid] = score
            uid_scores_dict[uid] = score
        else:
            scores[uid] = 0
            uid_scores_dict[uid] = 0
        wandb_data["scores"][uid] = score
        wandb_data["timestamps"][uid] = datetime.datetime.now().isoformat()

    if config.wandb_on: wandb.log(wandb_data)

    return scores, uid_scores_dict

async def query_text(dendrite, axon, uid, syn, config, subtensor, wallet):
    try:
        bt.logging.info(f"Sent query to uid: {uid}, {syn.messages[0]['content']} using {syn.engine}")
        full_response = ""
        responses = await asyncio.wait_for(dendrite([axon], syn, deserialize=False, streaming=True), 20)
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, list):
                    # bt.logging.info(chunk[0])
                    full_response += chunk[0]
            break
        return uid, full_response
    
    except Exception as e:
        bt.logging.error(f"Exception during query for uid {uid}: {e}\n{traceback.format_exc()}")
        return uid, None

async def get_and_score_text(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids):
    engine = "gpt-4-1106-preview"
    weight = 1
    seed = 1234

    # Data container for wandb logging
    wandb_data = {
        "prompts": {},
        "responses": {},
        "scores": {},
        "timestamps": {},
    }

    # Step 1: Query all text concurrently with associated questions
    query_tasks = []
    uid_to_question = {}
    for uid in available_uids:
        prompt = await get_question("text")
        uid_to_question[uid] = prompt
        messages = [{'role': 'user', 'content': prompt}]
        syn = StreamPrompting(messages=messages, engine=engine, seed=seed)
        task = query_text(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)
        query_tasks.append(task)

    query_responses = await asyncio.gather(*query_tasks)

    # log prompts and responses to wandb data
    for uid, response in query_responses:
        wandb_data["prompts"][uid] = uid_to_question[uid]
        wandb_data["responses"][uid] = response
        wandb_data["timestamps"][uid] = datetime.datetime.now().isoformat()

    # Step 2: Prepare all OpenAI call tasks for selected UIDs
    openai_tasks = []
    # Generate a random number once for the entire round
    random_number = random.random()
    will_score_all = random_number < 1/8

    bt.logging.info(f"Random Number: {random_number}, Will Score All: {will_score_all}")

    for uid, response in query_responses:
        # Decide to score all UIDs this round based on a 1/8 chance
        if will_score_all and response:
            messages = [{'role': 'user', 'content': uid_to_question[uid]}]
            task = call_openai(messages, 0, engine, seed)
            openai_tasks.append((uid, task))

    # Step 3: Run OpenAI calls concurrently for selected UIDs
    openai_responses = await asyncio.gather(*[task for _, task in openai_tasks])

    # Step 4: Prepare scoring tasks
    score_tasks = []
    for (uid, _), openai_answer in zip(openai_tasks, openai_responses):
        if openai_answer:
            question = uid_to_question[uid]
            response = next(res for u, res in query_responses if u == uid)  # Find the matching response
            task = template.reward.openai_score(openai_answer, response, weight)
            score_tasks.append((uid, task))

    # Step 5: Run scoring tasks concurrently
    scored_responses = await asyncio.gather(*[task for _, task in score_tasks])

    # Step 6: Update scores and uid_scores_dict
    for (uid, _), score in zip(score_tasks, scored_responses):
        if score is not None:
            scores[uid] = score
            uid_scores_dict[uid] = score
            # Log scores for UIDs that were scored
            wandb_data["scores"][uid] = score
        else:
            scores[uid] = 0
            uid_scores_dict[uid] = 0

    # Log data to wandb
    if config.wandb_on: wandb.log(wandb_data)

    return scores, uid_scores_dict
    
async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    steps_passed = 0
    total_scores = torch.zeros(len(metagraph.hotkeys))
    while True:
        try:
            # Sync metagraph and initialze scores
            metagraph = subtensor.metagraph(config.netuid)
            scores = torch.zeros(len(metagraph.hotkeys))
            uid_scores_dict = {}
            
            # Get the available UIDs
            available_uids = await get_available_uids(dendrite, metagraph)
            bt.logging.info(f"available_uids is {available_uids}")

            # # use text synapse 1/2 times
            # if steps_passed % 2 != 1:
            #     scores, uid_scores_dict = await get_and_score_text(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids)

            # else:
            scores, uid_scores_dict = await get_and_score_images(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids)

            total_scores += scores
            bt.logging.info(f"scores = {uid_scores_dict}, {2 - steps_passed % 3} iterations until set weights")
            bt.logging.info(f"total scores until set weights = {total_scores}")

            # Update weights after processing all batches
            if steps_passed % 5 == 4:
                bt.logging.info(f"total_scores = {total_scores}")
                avg_scores = total_scores / (steps_passed + 1)
                bt.logging.info(f"avg scores is {avg_scores}")
                steps_passed = 0
                set_weights(avg_scores, config, subtensor, wallet, metagraph)
                total_scores = torch.zeros(len(metagraph.hotkeys))

            steps_passed += 1
            time.sleep(2)

        except RuntimeError as e:
            bt.logging.error(f"RuntimeError: {e}\n{traceback.format_exc()}")
        except Exception as e:
            bt.logging.info(f"General exception: {e}\n{traceback.format_exc()}")

def main():
    global config
    config = get_config()
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    bt.logging.debug(f"got {wallet}, {subtensor}, {dendrite}, {metagraph}")
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    init_wandb(my_subnet_uid)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))
    return config

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        config = loop.run_until_complete(main())
    except KeyboardInterrupt:
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        template.utils.save_state_to_file(state)
        if config.wandb_on: wandb.finish()
    finally: loop.close()
