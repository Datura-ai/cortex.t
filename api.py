import bittensor as bt
import asyncio
import json
import traceback
from template.protocol import StreamPrompting, TextPrompting

# Assuming initial setup remains the same
wallet = bt.wallet( name="validator", hotkey="default" )
axon = bt.axon(wallet=wallet)
dendrite = bt.dendrite(wallet=wallet)
subtensor = bt.subtensor( network = "test")
metagraph = subtensor.metagraph(netuid = 24 )

# Simplified question setup
question = [{"role": "user", "content": "count to 20"}]
vali_uid = 1
target_uid = 1
provider = "OpenAI"
model = "gpt-3.5-turbo"
seed = 1234
temperature = 0.5
max_tokens = 2048
top_p = 0.8
top_k = 1000
timeout = 60
streaming = True

synapse = TextPrompting(
    messages=question,
    uid=target_uid,
    provider=provider,
    model=model,
    seed=seed,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    top_k=top_k,
    timeout=timeout,
    streaming=streaming,
)
bt.trace()
response = dendrite.query(metagraph.axons[target_uid], synapse)
print('completion', response.completion)

# async def query_miner(synapse):
#     try:
#         axon = metagraph.axons[vali_uid]
#         responses = dendrite.query(
#             axons=[axon], 
#             synapse=synapse, 
#             deserialize=False,
#             timeout=timeout,
#             streaming=streaming,
#         )
#         return await handle_response(responses)
#     except Exception as e:
#         print(f"Exception during query: {traceback.format_exc()}")
#         return None

# async def handle_response(responses):
#     full_response = ""
#     try:
#         for resp in responses:
#             async for chunk in resp:
#                 if isinstance(chunk, str):
#                     full_response += chunk  # Simplified for demonstration
#                     print(chunk)
#     except Exception as e:
#         print(f"Error processing response for uid {e}")
#     return full_response

# async def main():
#     response = await query_miner(synapse)
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())
