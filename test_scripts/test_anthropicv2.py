import json
import boto3
import sys


brt = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]

question = "tell me a fun fact"

body = json.dumps({
    'prompt': f'\n\nHuman: {question}\n\nAssistant:',
    'max_tokens_to_sample': 100,
    'temperature': 0.01,
    'top_p': 1,
})

for model in models:             
    response = brt.invoke_model_with_response_stream(
        modelId=model, 
        body=body
    )
        
    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk.get('bytes').decode())
                # Print only the completion part with flush=True
                print(chunk_data.get('completion'), end='', flush=True)
                sys.stdout.flush()
        print("\n")