import sys
sys.path.insert(0, '/root/bittensor/bittensor')

import bittensor as bt
from dendrite import * 
import os
import time
import torch
import argparse
import traceback
import template
import openai
import wandb
import re
import ast
import asyncio
import logging


# wandb.init(project="openai_qa", name="run1")

# Do this for the openai api key in terminal: echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc
openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

theme_counter = 0
question_counter = 0
themes = None
questions_list = None

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--custom", default="my_custom_value")
    parser.add_argument("--netuid", type=int, default=1)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
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

def call_openai(prompt, temperature, engine="gpt-3.5-turbo"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            temperature=0,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        bt.logging.info(f"Error when calling OpenAI: {e}")
        return None

def get_openai_answer(query, engine):
    temperature = 0
    answer = call_openai(query, temperature, engine)
    bt.logging.info(f"Response from openai: {answer}")
    return answer

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
        bt.logging.error(f"Error when extracting list: {e}")
        return None

def get_list_from_openai(prompt, default_list, max_retries=5):
    for retry_count in range(max_retries):
        try:
            answer = call_openai(prompt, .33).replace("\n", " ")
            # bt.logging.info(f"attempting to extract list from {answer}")
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
    # themes = get_list_from_openai(prompt, default_themes)
    themes = default_themes
    bt.logging.info(f"using themes of {themes}")
    return themes

def get_questions_list(theme):
    default_questions = ['What is the most important quality you look for in a partner?', 'How do you define love?', 'What is the most romantic gesture you have ever received?', 'What is your favorite love song and why?', 'What is the key to a successful long-term relationship?', 'What is your idea of a perfect date?', 'What is the best piece of relationship advice you have ever received?', 'What is the most memorable love story you have heard?', 'What is the biggest challenge in maintaining a healthy relationship?', 'What is your favorite way to show someone you love them?']
    prompt = f"Give me a python list of 10 different creative questions based off of the theme of {theme}. Max 15 words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete python list."
    # questions = get_list_from_openai(prompt, [])
    questions = default_questions
    return questions

def get_question():
    global theme_counter, question_counter, themes, questions_list

    if not themes:
        themes = get_themes()

    theme = themes[theme_counter]

    if not questions_list:
        questions_list = get_questions_list(theme)
        bt.logging.info(f"retrived new questions: {questions_list}")

    question = questions_list[question_counter]

    question_counter += 1
    if question_counter >= len(questions_list):
        questions_list = None
        question_counter = 0
        theme_counter += 1

        if theme_counter >= len(themes):
            themes = None
            theme_counter = 0

    return question


def set_weights(step, scores, config, subtensor, wallet, metagraph):
    weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
    bt.logging.info(f"weights is {weights}")

    result = subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=weights, wait_for_inclusion=True)
    if result:
        bt.logging.success("Successfully set weights.")
    else:
        bt.logging.error("Failed to set weights.")

def log_wandb(query, engine, responses_dict, step, timestamp):
    data = {
        '_timestamp': timestamp,
        '_runtime': time.time() - timestamp,
        'engine': engine,
        'prompt': query,
        '_step': step,
        'responses': []
    }

    for uid, response_data in responses_dict.items():
        response_entry = {
            'uid': uid,
            'response': response_data.get('response', None),
            'score': response_data.get('score', 0)
        }
        data['responses'].append(response_entry)

    wandb.log(data)

async def score_responses(query, engine, response_generators, config, scores):
    responses_dict = {}
    for i, synapse in enumerate(response_generators):
        full_response = []
        try:
            # Get the iterator from the synapse object
            chunks_iterator = synapse.stream_output_iter.__aiter__()

            # Wait for the first chunk with a timeout
            try:
                first_chunk = await asyncio.wait_for(chunks_iterator.__anext__(), timeout=0.2)
                full_response.append(first_chunk)
                async for chunk in chunks_iterator:
                    bt.logging.info(f"Received chunk: {chunk} from miner with UID {i}.")
                    full_response.append(chunk)
            except asyncio.TimeoutError:
                bt.logging.warning(f"Timeout while waiting for the first chunk from miner with UID {i}.")
                scores[i] = 0  # Assign a score of 0 due to timeout
                full_response = []

        except Exception as e:
            bt.logging.error(f"Error while processing chunks for miner with UID {i}: {e}")

        bt.logging.info(f"full response is {full_response}")
        full_response_str = ''.join(full_response)
        response_data = {"message": full_response_str}
        # openai_answer = get_openai_answer(query, engine)
        openai_answer = "yes"
        if openai_answer:
            score = template.reward.openai_score(openai_answer, full_response_str)
            bt.logging.info(f"Full response from miner with UID {i} scored: {score}")
            responses_dict[i] = {
                'response': full_response_str,
                'score': score
            }
            scores[i] = config.alpha * scores[i] + (1 - config.alpha) * score
        else:
            bt.logging.warning(f"no openai answer")

    bt.logging.info(f"scores = {scores}")
    return responses_dict

async def run_validator_loop(wallet, subtensor, dendrite, metagraph, config, scores):
    step = 0
    while True:
        try:
            bt.logging.info(f"Starting validator loop iteration {step}.")
            
            query = get_question()
            engine = "gpt-3.5"
            
            bt.logging.info(f"Sent query to miner: '{query}' using {engine}")
            
            # Create an empty list to store all chunks
            all_chunks = []
            
            # Query the dendrite and process the chunks as they arrive
            for chunk in dendrite.query(metagraph.axons, template.protocol.Openai(openai_input=query, openai_engine=engine), deserialize=True):
                bt.logging.info(f"Received chunk: {chunk}")
                all_chunks.append(chunk)
            
            # After processing all chunks, concatenate to get the full response
            full_response = ''.join(all_chunks)
            
            openai_answer = get_openai_answer(query, engine)
            if openai_answer:
                responses_dict = score_responses(openai_answer, [full_response], config, scores) # Note the [full_response] to make it a list.
                bt.logging.info(f"responses_dict is {responses_dict}")
                log_wandb(query, engine, responses_dict, step, time.time())

            if (step + 1) % 25 == 0:  
                set_weights(step, scores, config, subtensor, wallet, metagraph)

            bt.logging.info(f"step = {step}")
            step += 1
            metagraph = subtensor.metagraph(config.netuid)
            await asyncio.sleep(bt.__blocktime__)

        except RuntimeError as e:
            bt.logging.error(f"RuntimeError at step {step}: {e}")
        except Exception as e:
            logging.exception(f"General exception at step {step}: {e}")
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()

def main(config):
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    
    asyncio.run(run_validator_loop(wallet, subtensor, dendrite, metagraph, config, scores))

if __name__ == "__main__":
    main(get_config())