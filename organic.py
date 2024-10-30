import bittensor as bt
import asyncio
import traceback
import random
from cortext.dendrite import CortexDendrite
from cortext.protocol import StreamPrompting


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
        "wearable technology",
        "artificial neural networks",
        "drone technology",
        "digital currencies",
        "supercomputers",
        "genetic engineering",
        "precision medicine",
        "brain-computer interfaces",
        "quantum cryptography",
        "carbon capture technologies",
        "smart manufacturing",
        "biometrics",
        "edge computing",
        "cloud computing",
        "personalized healthcare",
        "cryptocurrency mining",
        "5G networks",
        "autonomous drones",
        "robot-assisted surgery",
        "big data analytics",
        "energy storage",
        "quantum supremacy",
        "self-driving trucks",
        "AI ethics",
        "distributed computing",
        "exoskeleton technology",
        "carbon-neutral technologies",
        "food security",
        "telemedicine",
        "smart grids",
        "renewable water resources",
    ]

    prompt_types = [
        "Explain the concept of {}.",
        "Discuss the potential impact of {}.",
        "Compare and contrast two approaches to {}.",
        "Outline the future prospects of {}.",
        "Describe the ethical implications of {}.",
        "Analyze the current state of {}.",
        "Propose a solution using {}.",
        "Evaluate the pros and cons of {}.",
        "Predict how {} will change in the next decade.",
        "Discuss the role of {} in solving global challenges.",
        "Explain how {} is transforming industry.",
        "Describe a day in the life with advanced {}.",
        "Outline the key challenges in developing {}.",
        "Discuss the intersection of {} and another field.",
        "Explain the historical development of {}.",
        "What are the key innovations in {}?",
        "How can {} be used to improve everyday life?",
        "What are the societal implications of {}?",
        "What are the business applications of {}?",
        "Discuss how {} could evolve in the next 50 years.",
        "What are the limitations of current {} technologies?",
        "How does {} compare to previous technologies?",
        "What are the most exciting breakthroughs in {}?",
        "What role does {} play in education?",
        "What are the security concerns related to {}?",
        "How can {} be integrated into future urban planning?",
        "What role will {} play in the future of transportation?",
        "How does {} affect personal privacy?",
        "Discuss the environmental impacts of {}.",
        "How is {} reshaping healthcare?",
        "What role does {} play in data security?",
        "What are the political implications of {}?",
        "What are the major regulatory hurdles for {}?",
        "What role will {} play in space exploration?",
        "How can {} address global inequality?",
        "Discuss the relationship between {} and creativity.",
        "How can {} be used in disaster management?",
        "What future careers might emerge due to {}?",
        "What is the relationship between {} and economics?",
        "How will {} impact future generations?",
        "How can governments promote the use of {}?",
    ]

    # Calculate the new maximum possible prompts
    max_possible_prompts = len(subjects) * len(prompt_types)
    num_prompts = min(num_prompts, max_possible_prompts)

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
    # wallet = bt.wallet(name="default", hotkey="default")
    wallet = bt.wallet(name="default", hotkey="default")
    dendrite = CortexDendrite(wallet=wallet)
    vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    axon_to_use = meta.axons[vali_uid]
    print(f"axon to use: {axon_to_use}")

    num_prompts = 50
    prompts = await generate_prompts(num_prompts)
    synapses = [StreamPrompting(
        messages=[{"role": "user", "content": prompt}],
        provider="OpenAI",
        model="gpt-4o"
    ) for prompt in prompts]
    # synapses = [StreamPrompting(
    #     messages=[{"role": "user", "content": prompt}],
    #     provider="Anthropic",
    #     model="claude-3-5-sonnet-20240620"
    # ) for prompt in prompts]
    # synapses = [StreamPrompting(
    #     messages=[{"role": "user", "content": prompt}],
    #     provider="Groq",
    #     model="llama-3.1-70b-versatile"
    # ) for prompt in prompts]

    async def query_and_log(synapse):
        return await query_miner(dendrite, axon_to_use, synapse)

    responses = await asyncio.gather(*[query_and_log(synapse) for synapse in synapses])

    import csv
    with open('miner_responses.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Prompt', 'Response'])
        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response])

    print("Responses saved to miner_responses.csv")


if __name__ == "__main__":
    asyncio.run(main())

