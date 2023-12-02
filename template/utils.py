import re
import os
import ast
import json
import wandb
import random
import asyncio
import template
import traceback
import bittensor as bt
from . import client

list_update_lock = asyncio.Lock()


def load_state_from_file(filename="state.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            bt.logging.info("loaded previous state")
            return json.load(file)
    else:
        bt.logging.info("initialized new global state")
        return {
            "text": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0},
            "images": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0}
        }

state = load_state_from_file()


def get_state():
    global state
    if state is None:
        load_state_from_file()
    return state


def save_state_to_file(state, filename="state.json"):
    with open(filename, "w") as file:
        bt.logging.success(f"saved global state to {filename}")
        json.dump(state, file)


def get_validators_with_runs_in_all_projects():
    api = wandb.Api()
    validators_runs = {project: set() for project in projects}

    # Retrieve runs for each project and store validator UIDs
    for project in template.PROJECT_NAMES:
        runs = api.runs(f"cortex-t/{project}")
        for run in runs:
            if run.config['type'] == 'validator':
                validators_runs[project].add(run.config['uid'])

    # Find common validators across all projects
    common_validators = set.intersection(*validators_runs.values())
    return common_validators


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
            extracted_list = extract_python_list(answer)
            if extracted_list:
                bt.logging.success(f"Received new {list_type}")
                bt.logging.debug(f"questions are {extracted_list}")
                return extracted_list
            else:
                bt.logging.info(f"No valid python list found, retry count: {retry + 1} {answer}")
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


def preprocess_string(text):
    try:
        text = text.replace("\t", " ")

        # Placeholder for single quotes within words
        placeholder = "___SINGLE_QUOTE___"

        # Replace single quotes within words with the placeholder
        processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, text)

        # Replace single quotes used for enclosing strings with double quotes
        processed_text = processed_text.replace("'", '"')

        # Restore the original single quotes from the placeholder
        processed_text = processed_text.replace(placeholder, "'")

        # Delete spaces after an opening bracket '['
        processed_text = re.sub(r"\[\s+", "[", processed_text)

        # Delete spaces before a closing bracket ']'
        processed_text = re.sub(r"\s+\]", "]", processed_text)

        # Remove characters before first '[' and after first ']'
        start = processed_text.find('[')
        end = processed_text.find(']')

        if start != -1 and end != -1 and end > start:
            processed_text = processed_text[start:end + 1]

        return processed_text

    except Exception as e:
        bt.logging.error(f"Error in preprocessing string: {traceback.format_exc()}")
        return text


def extract_python_list(text: str):
    try:
        text = preprocess_string(text)
        # Improved regex to match more complex list structures including multiline strings
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text)
        if match:
            list_str = match.group()

            # Using ast.literal_eval to safely evaluate the string as a Python literal
            evaluated = ast.literal_eval(list_str)
            if isinstance(evaluated, list):
                return evaluated
    except SyntaxError as e:
        bt.logging.error(f"Syntax error when extracting list: {e}\n{traceback.format_exc()}")
    except ValueError as e:
        bt.logging.error(f"Value error when extracting list: {e}\n{traceback.format_exc()}")
    except Exception as e:
        bt.logging.error(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    # Return None if the list cannot be extracted
    return None


async def call_openai(messages, temperature, model, seed=1234):
    for attempt in range(2):
        bt.logging.debug("Calling Openai ")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            response = response.choices[0].message.content
            bt.logging.debug(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {e}")
            await asyncio.sleep(0.5) 
    
    return None
