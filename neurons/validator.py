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
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument( '--wandb.on', action='store_true', help='Turn on wandb logging.')
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    args = parser.parse_args()
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    if config.wandb.on:
        run_name = f'validator-{my_uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
        config.uid = my_uid
        config.hotkey = wallet.hotkey.ss58_address
        config.run_name = run_name
        wandb_run =  wandb.init(
            name = run_name,
            anonymous = "allow",
            reinit = False,
            project = 'opentext_qa',
            entity = 'opentensor-dev',
            config = config,
            dir = config.full_path,
        )
        bt.logging.success( f'Started wandb run' )
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
    # bt.logging.info(f"Response from validator openai: {answer}")
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
        bt.logging.info(text)
        bt.logging.error(f"Error when extracting list: {e}")
        return None

def get_list_from_openai(prompt, default_list, max_retries=5):
    for retry_count in range(max_retries):
        try:
            answer = call_openai(prompt, .33).replace("\n", " ")
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
    themes = get_list_from_openai(prompt, default_themes)
    bt.logging.info(f"using themes of {themes}")
    return themes

def get_questions_list(theme):
    default_questions = ['What is the most important quality you look for in a partner?', 'How do you define love?', 'What is the most romantic gesture you have ever received?', 'What is your favorite love song and why?', 'What is the key to a successful long-term relationship?', 'What is your idea of a perfect date?', 'What is the best piece of relationship advice you have ever received?', 'What is the most memorable love story you have heard?', 'What is the biggest challenge in maintaining a healthy relationship?', 'What is your favorite way to show someone you love them?']
    prompt = f"Give me a python list of 10 different creative questions based off of the theme of {theme}. Max 15 words each. Provide it in python list structure and don't write anything extra, just provide exclusively the complete python list."
    questions = get_list_from_openai(prompt, [])
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


def set_weights(scores, config, subtensor, wallet, metagraph):
    bt.logging.info(f"weights is {scores}")
    subtensor.set_weights(netuid=config.netuid, wallet=wallet, uids=metagraph.uids, weights=scores, wait_for_inclusion=False)
    bt.logging.success("Successfully set weights.")

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


async def query_synapse(dendrite, metagraph, subtensor, config, wallet):
    step = 0
    while True:
        metagraph = subtensor.metagraph( 18 )
        available_uids = [ uid.item() for uid in metagraph.uids ]
        bt.logging.info(available_uids)
        uid = available_uids[169]
        scores = torch.zeros( len(metagraph.hotkeys) )
        # for uid in available_uids:
        try:
            axon = metagraph.axons[uid]
            response = dendrite.query(axon, IsAlive(), timeout = .5)
            if not response.is_success:
                bt.logging.info(f"failed response from uid: {uid}, axon: {axon}")
                time.sleep (.1)
                continue

            query = get_question()
            probability = random.random()
            engine = "gpt-4" if probability < 0.05 else "gpt-3.5-turbo"    
            bt.logging.info(f"Sent query to uid: {uid}, axon: {axon}, '{query}' using {engine}")
            syn = StreamPrompting(roles=["user"], messages=[query], engine = engine)

            async def main():
                full_response = ""
                responses = await dendrite([axon], syn, deserialize=False, streaming=True)
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
            full_response = await main()
            openai_answer = get_openai_answer(query, engine)
            score = template.reward.openai_score(openai_answer, full_response)
            scores[uid] = score
            bt.logging.info(f"score is {score}")
            # log_wandb(query, engine, responses_dict, step, time.time())

        except RuntimeError as e:
            bt.logging.error(f"RuntimeError at step {step}: {e}")
        except Exception as e:
            bt.logging.info(f"General exception at step {step}: {e}\n{traceback.format_exc()}")
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            if config.wandb.on: wandb_run.finish()
            exit()

    if (step + 1) % 3 == 0:  
        set_weights(scores, config, subtensor, wallet, metagraph)
    step += 1

def main(config):
    config = get_config()
    wallet, subtensor, dendrite, metagraph = initialize_components(config)
    check_validator_registration(wallet, subtensor, metagraph)
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
    asyncio.run(query_synapse(dendrite, metagraph, subtensor, config, wallet))

if __name__ == "__main__":
    main(get_config())