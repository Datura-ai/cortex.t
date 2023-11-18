import bittensor as bt
import os
import time
import torch
import argparse
import traceback
import template
from openai import OpenAI
import wandb
from typing import Optional, List
import random
import ast
import concurrent.futures
import asyncio
import string
from template.protocol import StreamPrompting, IsAlive, ImageResponse

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(timeout=30.0)

state = {
    "text": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0},
    "images": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0}
}

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on', help='Turn off wandb bt.logging.')
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    if config.wandb_on:
        run_name = f'validator-' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
        config.run_name = run_name
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
    return config

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

def call_openai(messages, temperature, engine, seed=1234):
    for attempt in range(3):
        bt.logging.info("Calling Openai")
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            response = response.choices[0].message.content
            bt.logging.debug(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI: {e}")
            time.sleep(0.5)
    
    return None

def extract_python_list(text: str) -> Optional[List]:
    """
    Extracts a Python list from a given string.
    Args:
        text (str): The string containing the Python list.
    Returns:
        Optional[List]: The extracted list if found and valid, otherwise None.
    """
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx == -1 or end_idx == -1:
            return None

        list_str = text[start_idx:end_idx+1]
        evaluated = ast.literal_eval(list_str)

        return evaluated if isinstance(evaluated, list) else None
    except Exception as e:
        bt.logging.info(text)
        bt.logging.error(f"Error when extracting list: {e}")
        return None

def get_list(list_type, theme=None):

    list_type_mapping = {
        "text_themes": {
            "default": template.question_themes,
            "prompt": "Create a Python list of 50 unique and thought-provoking themes, each suitable for generating meaningful text-based questions. Limit each theme to a maximum of four words. The themes should be diverse and encompass a range of topics, including technology, philosophy, society, history, science, and art. Format the themes as elements in a Python list, and provide only the list without any additional text or explanations."    
        },
        "images_themes": {
            "default": template.image_themes,
            "prompt": "Generate a Python list of 50 unique and broad creative themes for artistic inspiration. Each theme should be no more than four words, open to interpretation, and suitable for various artistic expressions. Present the list in a single-line Python list structure."
        },
        "text": {
            "default": template.text_questions,
            "prompt": f"Generate a Python list of 10 inventive and thought-provoking questions, each related to the theme '{theme}'. Ensure each question is concise, no more than 15 words, and tailored to evoke in-depth exploration or discussion about '{theme}'. Format the output as elements in a Python list, and include only the list without any additional explanations or text."
        },
        "images": {
            "default": template.image_questions,
            "prompt": f"Provide a list of 10 creative and detailed scenarios for image generation, each inspired by the theme '{theme}'. The scenarios should be diverse, encompassing elements such as natural landscapes, historical settings, futuristic scenes, and imaginative contexts related to '{theme}'. Each element in the list should be a concise but descriptive scenario, designed to inspire visually rich images. Format these as elements in a Python list."
        }
    }

     # Check if list_type is valid
    if list_type not in list_type_mapping:
        bt.logging.error("no valid list_type provided")
        return
    
    default = list_type_mapping[list_type]["default"]
    # prompt = list_type_mapping[list_type]["prompt"]

    # messages = [{'role': "user", 'content': prompt}]
    # max_retries = 3
    # for retry in range(max_retries):
    #     try:
    #         answer = call_openai(messages, .33, "gpt-3.5-turbo").replace("\n", " ")
    #         extracted_list = extract_python_list(answer)
    #         if extracted_list:
    #             bt.logging.info(f"Received {list_type}: {extracted_list}")
    #             return extracted_list
    #         else:
    #             bt.logging.info(f"No valid python list found, retry count: {retry + 1}")
    #     except Exception as e:
    #         retry += 1
    #         bt.logging.error(f"Got exception when calling openai {e}")

    # bt.logging.error(f"No list found after {max_retries} retries, using default list.")
    return default

def update_counters_and_get_new_list(category, item_type, theme=None):
    themes = state[category]["themes"]
    questions = state[category]["questions"]
    theme_counter = state[category]["theme_counter"]
    question_counter = state[category]["question_counter"]

    # Choose the appropriate counter and list based on item_type
    counter = theme_counter if item_type == "themes" else question_counter
    items = themes if item_type == "themes" else questions

    # Logic for updating counters and getting new list
    if not items:
        items = get_list(item_type, theme)
        counter = len(items) - 1  # Start at the end of the list

    item = items[counter]
    counter -= 1  # Move backwards in the list

    # Reset if we reach the front of the list
    if counter < 0:
        if item_type == "questions":
            questions = None
        else:  # item_type == "themes"
            themes = None
            theme_counter -= 1  # Move to the previous theme

    # Update the state
    state[category]["themes"] = themes
    state[category]["questions"] = questions
    state[category]["theme_counter"] = theme_counter
    state[category]["question_counter"] = question_counter

    return item

def get_question(category):
    if category not in ["text", "images"]:
        raise ValueError("Invalid category. Must be 'text' or 'images'.")

    theme = update_counters_and_get_new_list(category, f"{category}_themes")
    question = update_counters_and_get_new_list(category, f"{category}", theme)

    return question

def log_wandb(query, engine, responses):
    data = {
        '_timestamp': time.time(),
        'engine': engine,
        'prompt': query,
        'responses': responses
    }

    wandb.log(data)

async def query_miner(dendrite, axon, uid, syn, config, subtensor, wallet):
    try:
        bt.logging.info(f"Sent query to uid: {uid}, {syn.messages} using {syn.engine}")
        full_response = ""
        responses = await asyncio.wait_for(dendrite([axon], syn, deserialize=False, streaming=True), 50)
        for resp in responses:
            i = 0
            async for chunk in resp:
                i += 1
                if isinstance(chunk, list):
                    print(chunk[0], end="", flush=True)
                    full_response += chunk[0]
                else:
                    synapse = chunk
            break
        print("\n")
        return full_response
    
    except Exception as e:
        bt.logging.error(f"Exception during query for uid {uid}: {e}")

async def check_uid(dendrite, axon, uid):
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=.1)
        if response.is_success:
            bt.logging.info(f"UID {uid} is active")
            return uid
        else:
            bt.logging.info(f"UID {uid} is not active")
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
    alpha = .75
    if moving_average_scores is None:
        moving_average_scores = scores.clone()

    # Update the moving average scores
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores
    bt.logging.info(f"Updated moving average of weights: {moving_average_scores}")
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=moving_average_scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights based on moving average.")

async def query_image(dendrite, axon, uid, syn, config, subtensor, wallet):
    try:
        bt.logging.info(f"Sent image request to uid: {uid}, {syn.messages} using {syn.engine}")
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

    # Step 1: Query all images concurrently
    query_tasks = []
    for uid in available_uids:
        messages = get_question("images")
        syn = ImageResponse(messages=messages, engine=engine, size=size, quality=quality, style=style)
        task = query_image(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)
        query_tasks.append(task)

    query_responses = await asyncio.gather(*query_tasks)

    # Step 2: Score all images concurrently
    score_tasks = []
    for i, (uid, response) in enumerate(query_responses):
        if response:
            response = response[0]
            completion = response.completion
            bt.logging.info(f"response for uid {i} is {completion}")
            url = completion["url"]
            score = await template.reward.image_score(url, size, messages, weight)
            scores[i] = score  # Assign score to the tensor
            uid_scores_dict[uid] = score
        else:
            # Handle no response
            scores[i] = 0  # Assign default score to the tensor
            uid_scores_dict[uid] = 0

    return scores, uid_scores_dict

    scored_responses = await asyncio.gather(*score_tasks)

    # Step 3: Update scores and uid_scores_dict
    for (uid, _), score in zip(query_responses, scored_responses):
        if score is not None:
            scores[i] = score
            uid_scores_dict[uid] = score

    return scores, uid_scores_dict
    
async def get_and_score_text(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids):
    # engine = "gpt-4-1106-preview"
    engine = "gpt-3.5-turbo"
    weight = 1
    seed=1234
    for i in range(len(available_uids)):
        uid = available_uids[i]

        # Get new questions
        prompt = get_question("text")
        messages = [{'role': 'user', 'content': prompt}]
        syn = StreamPrompting(messages=messages, engine=engine, seed=seed)

        # Query miners
        task = [query_miner(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)]
        response = await asyncio.gather(*task)

        # Get OpenAI answer for the current batch
        openai_answer = call_openai(messages, 0, engine, seed)

        # Calculate scores for each response in the current batch
        if openai_answer:
            score = [template.reward.openai_score(openai_answer, response, weight)]
            # Update the scores array with batch scores at the correct indices
            scores[uid] = score
            uid_scores_dict[uid] = score

            if config.wandb_on:
                log_wandb(query, engine, responses)

    return scores, uid_scores_dict
    
async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    step_counter = 0
    steps_passed = 0
    while True:
        try:
            metagraph = subtensor.metagraph(24)
            total_scores = torch.zeros(len(metagraph.hotkeys))
            scores = torch.zeros(len(metagraph.hotkeys))
            uid_scores_dict = {}
            
            # Get the available UIDs
            available_uids = await get_available_uids(dendrite, metagraph)
            # available_uids = [2]
            bt.logging.info(f"available_uids is {available_uids}")

            # # use text synapse 3/4 times
            # if step_counter % 4 != 3:
            # scores, uid_scores_dict = await get_and_score_text(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids)

            # else:
            scores, uid_scores_dict = await get_and_score_images(dendrite, metagraph, config, subtensor, wallet, scores, uid_scores_dict, available_uids)
            time.sleep(8)
            total_scores += scores
            bt.logging.info(f"scores = {uid_scores_dict}, {3 - step_counter % 3} iterations until set weights")

            # Update weights after processing all batches
            if steps_passed % 3 == 2:
                avg_scores = total_scores / steps_passed
                bt.logging.info(f"avg scores is {avg_scores}")
                steps_passed = 0
                set_weights(avg_scores, config, subtensor, wallet, metagraph)

            step_counter += 1

        except RuntimeError as e:
            bt.logging.error(f"RuntimeError: {e}\n{traceback.format_exc()}")
        except Exception as e:
            bt.logging.info(f"General exception: {e}\n{traceback.format_exc()}")
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            if config.wandb_on: wandb.finish()
            exit()


def main():
    config = get_config()
    bt.logging.debug(f"got config  {config}")
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    bt.logging.debug(f"got  {wallet}, {subtensor}, {dendrite}, {metagraph}")
    check_validator_registration(wallet, subtensor, metagraph)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))

if __name__ == "__main__":
    main()