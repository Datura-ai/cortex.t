import bittensor as bt
import asyncio
import random
import traceback
from cortext.protocol import StreamPrompting
from cortext.dendrite import CortexDendrite
import aiohttp
from validators.services.cache import cache_service



def load_entire_questions():
    # Asynchronous function to fetch a URL
    async def fetch(session, url):
        async with session.get(url) as response:
            try:
                return await response.json()
            except Exception as err:
                bt.logging.error(f"{err} {traceback.format_exc()}")

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

async def generate_prompts(num_prompts=100):
    subjects = [
        "artificial intelligence",
        "climate change",
        "space exploration",
        "quantum computing",
        "renewable energy",
        "virtual reality",
        "biotechnology",
        "cybersecurity",
        "autonomous vehicles",
        "blockchain",
        "3D printing",
        "robotics",
        "nanotechnology",
        "gene editing",
        "Internet of Things",
        "augmented reality",
        "machine learning",
        "sustainable agriculture",
        "smart cities",
        "digital privacy",
    ]

    prompt_types = [
        "Explain the concept of",
        "Discuss the potential impact of",
        "Compare and contrast two approaches to",
        "Outline the future prospects of",
        "Describe the ethical implications of",
        "Analyze the current state of",
        "Propose a solution using",
        "Evaluate the pros and cons of",
        "Predict how {} will change in the next decade",
        "Discuss the role of {} in solving global challenges",
        "Explain how {} is transforming industry",
        "Describe a day in the life with advanced {}",
        "Outline the key challenges in developing {}",
        "Discuss the intersection of {} and another field",
        "Explain the historical development of",
    ]

    prompts = set()
    while len(prompts) < num_prompts:
        subject = random.choice(subjects)
        prompt_type = random.choice(prompt_types)
        prompt = prompt_type.format(subject)
        prompts.add(prompt)

    return list(prompts)


async def query_miner(dendrite: CortexDendrite, axon_to_use, synapse, timeout=60, streaming=True):
    try:
        print(f"calling vali axon {axon_to_use} to miner uid {synapse.uid} for query {synapse.messages}")
        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=timeout
        )
        return await handle_response(resp)
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        return None


async def handle_response(resp):
    full_response = ""
    try:
        async for chunk in resp:
            if isinstance(chunk, str):
                full_response += chunk
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
    except Exception as e:
        print(f"Error processing response for uid {e}")
    return full_response


async def main():
    print("synching metagraph, this takes way too long.........")
    subtensor = bt.subtensor(network="finney")
    meta = subtensor.metagraph(netuid=18)
    print("metagraph synched!")

    # This needs to be your validator wallet that is running your subnet 18 validator
    wallet = bt.wallet(name="default", hotkey="default")
    dendrite = CortexDendrite(wallet=wallet)
    vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    axon_to_use = meta.axons[vali_uid]
    print(f"axon to use: {axon_to_use}")

    num_prompts = 10
    prompts = load_entire_questions()
    prompts = prompts[:2]
    synapses = [StreamPrompting(
        messages=[{"role": "user", "content": prompt}],
        provider="OpenAI",
        model="gpt-4o"
    ) for prompt in prompts]

    an_synapses = [StreamPrompting(
        messages=[{"role": "user", "content": prompt}],
        provider="Anthropic",
        model="claude-3-5-sonnet-20240620"
    ) for prompt in prompts]
    synapses += an_synapses

    async def query_and_log(synapse):
        return await query_miner(dendrite, axon_to_use, synapse)

    responses = await asyncio.gather(*[query_and_log(synapse) for synapse in synapses])

    cache_service.set_cache_in_batch(synapses)

    print("Responses saved to cache database")


if __name__ == "__main__":
    asyncio.run(main())

