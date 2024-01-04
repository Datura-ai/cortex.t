import json
import boto3
import sys
import asyncio

# async def fetch_model_responses(question, model):
#     brt = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

#     body = json.dumps({
#         'prompt': f'\n\nHuman: {question}\n\nAssistant:',
#         'max_tokens_to_sample': 100,
#         'temperature': 0.01,
#         'top_p': 1,
#     })

#     loop = asyncio.get_running_loop()

#     response = await loop.run_in_executor(
#         None, 
#         lambda: brt.invoke_model_with_response_stream(modelId=model, body=body)
#     )
#     stream = response.get('body')
#     if stream:
#         for event in stream:
#             chunk = event.get('chunk')
#             if chunk:
#                 chunk_data = json.loads(chunk.get('bytes').decode())
#                 print(chunk_data.get('completion'), end='', flush=True)
#                 sys.stdout.flush()
#         print("\n")

# # Example usage
# models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]
# model = models[1]
# question = "tell me a short story"
# asyncio.run(fetch_model_responses(question, model))

import anthropic_bedrock
from anthropic_bedrock import AsyncAnthropicBedrock

client = AsyncAnthropicBedrock()


async def call_anthropic(question, model, max_tokens):
    completion = await client.completions.create(
        model=model,
        max_tokens_to_sample=max_tokens,
        prompt=f"{anthropic_bedrock.HUMAN_PROMPT} {question} {anthropic_bedrock.AI_PROMPT}",
    )
    print(completion.completion)
    return completion.completion

models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]
model = models[1]
question = "tell me a short story"
max_tokens = 2048

asyncio.run(call_anthropic(question, model, max_tokens))
