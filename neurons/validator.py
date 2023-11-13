import bittensor as bt
import os
import time
import torch
import argparse
import traceback
import template
import openai
import wandb
import re
import random
import ast
import asyncio
from template.protocol import StreamPrompting, IsAlive
import string

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

theme_counter = 0
question_counter = 0
themes = None
questions_list = None
moving_average_scores = None

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
    max_retries = 5
    for attempt in range(max_retries):
        bt.logging.info("Calling Openai")
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(.5)

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

def get_list_from_openai(prompt, default_list, max_retries=5):
    for retry_count in range(max_retries):
        try:
            answer = call_openai(prompt, .33, "gpt-3.5-turbo").replace("\n", " ")
            extracted_list = extract_python_list(answer)
            if extracted_list:
                return extracted_list
            else:
                bt.logging.info(f"No valid python list found, retry count: {retry_count + 1}")
        except Exception as e:
            bt.logging.error(f"Got exception when calling openai {e}")
    else:
        bt.logging.error(f"No list found after {max_retries} retries, using default list.")
        return default_list

def get_themes(num_themes=50):
    default_themes = ['Love and relationships', 'Nature and environment', 'Art and creativity', 'Technology and innovation', 'Health and wellness', 'History and culture', 'Science and discovery', 'Philosophy and ethics', 'Education and learning', 'Music and rhythm', 'Sports and athleticism', 'Food and nutrition', 'Travel and adventure', 'Fashion and style', 'Books and literature', 'Movies and entertainment', 'Politics and governance', 'Business and entrepreneurship', 'Mind and consciousness', 'Family and parenting', 'Social media and networking', 'Religion and spirituality', 'Money and finance', 'Language and communication', 'Human behavior and psychology', 'Space and astronomy', 'Climate change and sustainability', 'Dreams and aspirations', 'Equality and social justice', 'Gaming and virtual reality', 'Artificial intelligence and robotics', 'Creativity and imagination', 'Emotions and feelings', 'Healthcare and medicine', 'Sportsmanship and teamwork', 'Cuisine and gastronomy', 'Historical events and figures', 'Scientific advancements', 'Ethical dilemmas and decision making', 'Learning and growth', 'Music genres and artists', 'Film genres and directors', 'Government policies and laws', 'Startups and innovation', 'Consciousness and perception', 'Parenting styles and techniques', 'Online communities and forums', 'Religious practices and rituals', 'Personal finance and budgeting', 'Linguistic diversity and evolution', 'Human cognition and memory', 'Astrology and horoscopes', 'Environmental conservation', 'Personal development and self-improvement', 'Sports strategies and tactics', 'Culinary traditions and customs', 'Ancient civilizations and empires', 'Medical breakthroughs and treatments', 'Moral values and principles', 'Critical thinking and problem solving', 'Musical instruments and techniques', 'Film production and cinematography', 'International relations and diplomacy', 'Corporate culture and work-life balance', 'Neuroscience and brain function', 'Childhood development and milestones', 'Online privacy and cybersecurity', 'Religious tolerance and understanding', 'Investment strategies and tips', 'Language acquisition and fluency', 'Social influence and conformity', 'Space exploration and colonization', 'Sustainable living and eco-friendly practices', 'Self-reflection and introspection', 'Sports psychology and mental training', 'Globalization and cultural exchange', 'Political ideologies and systems', 'Entrepreneurial mindset and success', 'Conscious living and mindfulness', 'Positive psychology and happiness', 'Music therapy and healing', 'Film analysis and interpretation', 'Human rights and advocacy', 'Financial literacy and money management', 'Multilingualism and translation', 'Social media impact on society', 'Religious extremism and radicalization', 'Real estate investment and trends', 'Language preservation and revitalization', 'Social inequality and discrimination', 'Climate change mitigation strategies', 'Self-care and well-being', 'Sports injuries and rehabilitation', 'Artificial intelligence ethics', 'Creativity in problem solving', 'Emotional intelligence and empathy', 'Healthcare access and affordability', 'Sports analytics and data science', 'Cultural appropriation and appreciation', 'Ethical implications of technology']
    prompt = f"Give me a python list of {num_themes} different creative themes of which one could ask meaningful questions. Max four words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete list."
    messages = [{'role': "user", 'content': prompt}]
    themes = get_list_from_openai(messages, default_themes)
    bt.logging.info(f"using themes of {themes}")
    return themes

def get_questions_list(theme):
    default_questions = ['What is the most important quality you look for in a partner?', 'How do you define love?', 'What is the most romantic gesture you have ever received?', 'What is your favorite love song and why?', 'What is the key to a successful long-term relationship?', 'What is your idea of a perfect date?', 'What is the best piece of relationship advice you have ever received?', 'What is the most memorable love story you have heard?', 'What is the biggest challenge in maintaining a healthy relationship?', 'What is your favorite way to show someone you love them?']
    prompt = f"Give me a python list of 10 different creative questions based off of the theme of {theme}. Max 15 words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete python list."
    messages = [{'role': "user", 'content': prompt}]
    questions = get_list_from_openai(messages, default_questions)
    return questions

def get_question():
    global theme_counter, question_counter, themes, questions_list

    if not themes:
        themes = get_themes()
        theme_counter = len(themes) - 1  # Start at the end of the themes list

    theme = themes[theme_counter]

    if not questions_list:
        questions_list = get_questions_list(theme)
        question_counter = len(questions_list) - 1  # Start at the end of the questions list
        bt.logging.info(f"retrieved new questions: {questions_list}")

    question = questions_list[question_counter]

    # Move backwards in the questions list
    question_counter -= 1
    if question_counter < 0:  # If we reach the front, get new questions
        questions_list = None
        theme_counter -= 1  # Move to the previous theme

        if theme_counter < 0:  # If we reach the front of themes, start over
            themes = None

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
    alpha = .85

    # First time, initialize the moving average scores with the current scores
    if moving_average_scores is None:
        moving_average_scores = scores.clone()

    # Update the moving average scores
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores

    # Logging the updated moving average
    bt.logging.info(f"Updated moving average of weights: {moving_average_scores}")

    # Set the weights based on the moving average
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=moving_average_scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights based on moving average.")
    
async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    step_counter = 0  # Counter to track when to switch engines
    total_scores = torch.zeros(len(metagraph.hotkeys))
    steps_passed = 0
    while True:
        try:
            # Determine the engine based on the counter
            if step_counter % 5 == 4:  # Use gpt-4 every fourth iteration
                engine = "gpt-4"
                weight = 1
            else:
                engine = "gpt-3.5-turbo"
                weight = 0.7

            metagraph = subtensor.metagraph(18)
            scores = torch.zeros(len(metagraph.hotkeys))
            uid_scores_dict = {}
            
            # Get the available UIDs
            available_uids = get_available_uids(dendrite, metagraph)
            bt.logging.info(f"available_uids is {available_uids}")

            # Process in batches of 10 UIDs
            batch_size = 10
            for i in range(0, len(available_uids), batch_size):
                current_batch = available_uids[i:i+batch_size]

                # Get a new question for each batch
                query = get_question()
                role = "user"
                messages = [{'role': role, 'content': query}]

                # Create StreamPrompting object with the selected engine
                syn = StreamPrompting(messages=messages, engine=engine)

                # Query miners
                tasks = [query_miner(dendrite, metagraph.axons[uid], uid, syn, config, subtensor, wallet) for uid in current_batch]
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

                    total_scores += scores
                    bt.logging.info(f"scores = {uid_scores_dict}, {2 - steps_passed} iterations until set weights")

                    if config.wandb_on:
                        log_wandb(query, engine, responses)

            steps_passed += 1
            # Update weights after processing all batches
            if step_counter % 3 == 2:
                avg_scores = total_scores / steps_passed
                bt.logging.info(f"avg scores is {avg_scores}")
                total_scores = torch.zeros_like(scores)
                steps_passed = 0
                set_weights(avg_scores, config, subtensor, wallet, metagraph)
            step_counter += 1

        except RuntimeError as e:
            bt.logging.error(f"RuntimeError: {e}\n{traceback.format_exc()}")
        except Exception as e:
            bt.logging.info(f"General exception: {e}\n{traceback.format_exc()}")
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            if config.wandb_on: wandb_run.finish()
            exit()


def main(config):
    config = get_config()
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))

if __name__ == "__main__":
    main(get_config())