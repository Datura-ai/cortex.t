import os
import random
import asyncio
import traceback
import numpy as np
import bittensor as bt
from openai import AsyncOpenAI
from datasets import load_dataset

AsyncOpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not AsyncOpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=30.0)

async def embeddings(texts, model):
    bt.logging.info(f"Received embeddings request: {texts, model}")

    async def get_embeddings_in_batch(texts, model, batch_size=10):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []
        for batch in batches:
            filtered_batch = [text for text in batch if text.strip()]

            if filtered_batch:
                response = await client.embeddings.create(input=filtered_batch, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            else:
                bt.logging.info("Skipped an empty batch.")
        return all_embeddings


    try:
        # Assuming 'client' is already defined and authenticated
        batched_embeddings = await get_embeddings_in_batch(texts, model=model)
        embeddings = [np.array(embed) for embed in batched_embeddings]

        return embeddings
    except Exception as e:
        bt.logging.error(f"Exception in embeddings function: {traceback.format_exc()}")


def get_random_texts(dataset_name, config_name, num_samples=100):
    dataset = load_dataset(dataset_name, config_name)
    texts = [item['text'] for item in dataset['train']] 
    return random.sample(texts, num_samples)

random_texts = get_random_texts('wikitext', 'wikitext-2-v1', 100)
print(random_texts)

def main():
    # Run the async function and get the event loop
    loop = asyncio.get_event_loop()
    embeddings_result = loop.run_until_complete(embeddings(random_texts, "text-embedding-ada-002"))
    return embeddings_result

# Call the main function to run the async code
embeddings_result = main()
print(embeddings_result)
