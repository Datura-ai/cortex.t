import bittensor as bt
import os
import time
import torch
import argparse
import traceback
import template
from openai import OpenAI
import wandb
import re
import random
import ast
import asyncio
from template.protocol import StreamPrompting, IsAlive
import string

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
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on', help='Turn off wandb logging.')
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
        bt.logging.error(f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again.")
        exit()

def call_openai(messages, temperature, engine):
    for attempt in range(3):
        bt.logging.info("Calling Openai")
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                seed=1234,
            )
            bt.logging.debug(f"validator response is {response}")

            return response['choices'][0]['message']['content']

        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI: {e}")
            time.sleep(.5)
    
    return None

def extract_python_list(text):
    try:
        # Find the first open bracket and the last closing bracket
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx == -1 or end_idx == -1:
            return None

        list_str = text[start_idx:end_idx+1]
        evaluated = ast.literal_eval(list_str)

        if isinstance(evaluated, list):
            return evaluated
        return None
    except Exception as e:
        bt.logging.info(text)
        bt.logging.error(f"Error when extracting list: {e}")
        return None

def get_list(list_type, theme=None):
    if list_type == "questions":
        default = ['What is the most important quality you look for in a partner?', 'How do you define love?', 'What is the most romantic gesture you have ever received?', 'What is your favorite love song and why?', 'What is the key to a successful long-term relationship?', 'What is your idea of a perfect date?', 'What is the best piece of relationship advice you have ever received?', 'What is the most memorable love story you have heard?', 'What is the biggest challenge in maintaining a healthy relationship?', 'What is your favorite way to show someone you love them?']
        prompt = f"Give me a python list of 10 different creative questions based off of the theme of {theme}. Max 15 words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete python list."}]

    elif list_type == "themes":
        default = ['Love and relationships', 'Nature and environment', 'Art and creativity', 'Technology and innovation', 'Health and wellness', 'History and culture', 'Science and discovery', 'Philosophy and ethics', 'Education and learning', 'Music and rhythm', 'Sports and athleticism', 'Food and nutrition', 'Travel and adventure', 'Fashion and style', 'Books and literature', 'Movies and entertainment', 'Politics and governance', 'Business and entrepreneurship', 'Mind and consciousness', 'Family and parenting', 'Social media and networking', 'Religion and spirituality', 'Money and finance', 'Language and communication', 'Human behavior and psychology', 'Space and astronomy', 'Climate change and sustainability', 'Dreams and aspirations', 'Equality and social justice', 'Gaming and virtual reality', 'Artificial intelligence and robotics', 'Creativity and imagination', 'Emotions and feelings', 'Healthcare and medicine', 'Sportsmanship and teamwork', 'Cuisine and gastronomy', 'Historical events and figures', 'Scientific advancements', 'Ethical dilemmas and decision making', 'Learning and growth', 'Music genres and artists', 'Film genres and directors', 'Government policies and laws', 'Startups and innovation', 'Consciousness and perception', 'Parenting styles and techniques', 'Online communities and forums', 'Religious practices and rituals', 'Personal finance and budgeting', 'Linguistic diversity and evolution', 'Human cognition and memory', 'Astrology and horoscopes', 'Environmental conservation', 'Personal development and self-improvement', 'Sports strategies and tactics', 'Culinary traditions and customs', 'Ancient civilizations and empires', 'Medical breakthroughs and treatments', 'Moral values and principles', 'Critical thinking and problem solving', 'Musical instruments and techniques', 'Film production and cinematography', 'International relations and diplomacy', 'Corporate culture and work-life balance', 'Neuroscience and brain function', 'Childhood development and milestones', 'Online privacy and cybersecurity', 'Religious tolerance and understanding', 'Investment strategies and tips', 'Language acquisition and fluency', 'Social influence and conformity', 'Space exploration and colonization', 'Sustainable living and eco-friendly practices', 'Self-reflection and introspection', 'Sports psychology and mental training', 'Globalization and cultural exchange', 'Political ideologies and systems', 'Entrepreneurial mindset and success', 'Conscious living and mindfulness', 'Positive psychology and happiness', 'Music therapy and healing', 'Film analysis and interpretation', 'Human rights and advocacy', 'Financial literacy and money management', 'Multilingualism and translation', 'Social media impact on society', 'Religious extremism and radicalization', 'Real estate investment and trends', 'Language preservation and revitalization', 'Social inequality and discrimination', 'Climate change mitigation strategies', 'Self-care and well-being', 'Sports injuries and rehabilitation', 'Artificial intelligence ethics', 'Creativity in problem solving', 'Emotional intelligence and empathy', 'Healthcare access and affordability', 'Sports analytics and data science', 'Cultural appropriation and appreciation', 'Ethical implications of technology']
        prompt = f"Give me a python list of {num_themes} different creative themes of which one could ask meaningful questions. Max four words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete list."}]
    
    elif list_type == "images":
        default = ['A majestic golden eagle soaring high above a mountain range, its powerful wings spread wide against a clear blue sky.', 'A bustling medieval marketplace, full of colorful stalls, various goods, and people dressed in period attire, with a castle in the background.', 'An underwater scene showcasing a vibrant coral reef teeming with diverse marine life, including fish, sea turtles, and starfish.', 'A serene Zen garden with neatly raked sand, smooth stones, and a small, gently babbling brook surrounded by lush green foliage.', 'A futuristic cityscape at night, illuminated by neon lights, with flying cars zooming between towering skyscrapers.', 'A cozy cabin in a snowy forest at twilight, with warm light glowing from the windows and smoke rising from the chimney.', 'A surreal landscape with floating islands, cascading waterfalls, and a path leading to a castle in the sky, set against a sunset backdrop.', 'An astronaut exploring the surface of Mars, with a detailed spacesuit, the red Martian terrain around, and Earth visible in the sky.', 'A lively carnival scene with a Ferris wheel, colorful tents, crowds of happy people, and the air filled with the smell of popcorn and cotton candy.', 'A majestic lion resting on a savanna, with the African sunset in the background, highlighting its powerful mane and serene expression.']
        prompt = "Provide a list of 10 creative and detailed scenarios for image generation, formatted as elements in a Python list. The scenarios should be diverse, encompassing themes such as natural landscapes, historical settings, futuristic scenes, and other imaginative contexts. Each element in the list should be a concise but descriptive scenario, designed to inspire visually rich images. Provide exclusively the python list."

    else:
        bt.logging.error("no valid list_type provided")

    messages = [{'role': "user", 'content': prompt}]
    for retry in range(3):
        try:
            answer = call_openai(prompt, .33, "gpt-3.5-turbo").replace("\n", " ")
            extracted_list = extract_python_list(answer)
            if extracted_list:
                bt.logging.info(f"using list of type {list_type}: {full_list}")
                return extracted_list
            else:
                bt.logging.info(f"No valid python list found, retry count: {retry + 1}")
        except Exception as e:
            bt.logging.error(f"Got exception when calling openai {e}")
    else:
        bt.logging.error(f"No list found after {max_retries} retries, using default list.")
        return default

def update_counters_and_get_new_list(category, item_type):
    themes = state[category]["themes"]
    questions = state[category]["questions"]
    theme_counter = state[category]["theme_counter"]
    question_counter = state[category]["question_counter"]

    # Choose the appropriate counter and list based on item_type
    counter = theme_counter if item_type == "themes" else question_counter
    items = themes if item_type == "themes" else questions

    # Logic for updating counters and getting new list
    if not items:
        items = get_list(item_type, theme, category)
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

    theme = update_counters_and_get_new_list(category, "themes")
    question = update_counters_and_get_new_list(category, "questions")

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
        bt.logging.info(f"Sent query to uid: {uid}, '{syn.messages}' using {syn.engine}")
        full_response = ""
        responses = await asyncio.wait_for(dendrite([axon], syn, deserialize=False, streaming=True), 5)
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

def get_available_uids(dendrite, metagraph):
    available_uids = []
    for uid in metagraph.uids:
        axon = metagraph.axons[uid.item()]
        response = dendrite.query(axon, IsAlive(), timeout=1)
        if response.is_success:
            bt.logging.info(f"UID {uid.item()} is active")
            available_uids.append(uid.item())
        else:
            bt.logging.info(f"UID {uid.item()} is not active")

    return available_uids

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

def get_and_score_image(dendrite, metagraph, config, subtensor, wallet):
    engine = "dall-e-3"
    weight = 1
    size = "1024x1024"
    quality = "standard"
    style = "vivid"

    for i in range(len(available_uids):
        uid = available_uids[i]

        # Get new questions
        messages = get_question("images")
        syn = ImageResponse(messages=messages, engine=engine weight=weight, size=size, quality=quality, style=style)

        # Query miners
        tasks = [query_miner(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)]
        responses = await asyncio.gather(*tasks)
    

def get_and_score_text(dendrite, metagraph, config, subtensor, wallet):
    engine = "gpt-4-1106-preview"
    weight = 1

    for i in range(len(available_uids):
        uid = available_uids[i]

        # Get new questions
        prompt = get_question("text")
        messages = [{'role': 'user', 'content': prompt}]
        syn = StreamPrompting(messages=messages, engine=engine)

        # Query miners
        tasks = [query_miner(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet)]
        responses = await asyncio.gather(*tasks)

        # Get OpenAI answer for the current batch
        openai_answer = call_openai(messages, 0, engine)

        # Calculate scores for each response in the current batch
        if openai_answer:
            batch_scores = [template.reward.openai_score(openai_answer, response, weight) for response in responses]
            # Update the scores array with batch scores at the correct indices
            for uid, score in zip(current_batch, batch_scores):
                scores[uid] = score
                uid_scores_dict[uid] = score

            if config.wandb_on:
                log_wandb(query, engine, responses)

    return scores, uid_scores_dict

def get_and_score_text():
    engine = "dall-e-3"
    weight = 1



    
async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    step_counter = 0
    while True:
        try:
            metagraph = subtensor.metagraph(18)
            total_scores = torch.zeros(len(metagraph.hotkeys))
            scores = torch.zeros(len(metagraph.hotkeys))
            uid_scores_dict = {}
            
            # Get the available UIDs
            available_uids = get_available_uids(dendrite, metagraph)
            bt.logging.info(f"available_uids is {available_uids}")

            # use text synapse 3/4 times
            if counter % 4 != 3:
                scores, uid_scores_dict = get_and_score_text(scores, uid_scores_dict)

            elif counter % 4 == 3:
                get_and_score_image()

            total_scores += scores
            bt.logging.info(f"scores = {uid_scores_dict}, {2 - steps_passed} iterations until set weights")

            # Update weights after processing all batches
            if step_counter % 3 == 2:
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


def main(config):
    config = get_config()
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))

if __name__ == "__main__":
    main(get_config())