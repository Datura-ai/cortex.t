import asyncio
import os
import traceback
import json
import aioboto3

bedrock_client_parameters = {
    "service_name": 'bedrock-runtime',
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY"),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_KEY"),
    "region_name": "us-east-1"
}

async def send_bedrock_request(prompt, model="cohere.command-r-v1:0"):
    try:
        aws_session = aioboto3.Session()
        bedrock_client = aws_session.client(**bedrock_client_parameters)

        native_request = {
            "message": prompt,
            "temperature": 0.01,
            "seed": 1234,
        }

        async with bedrock_client as client:
            request = json.dumps(native_request)
            response = await client.invoke_model(
                modelId=model, body=request
            )

            response_body = await response['body'].read()
            response_body = json.loads(response_body)
            print(response_body)

    except Exception as e:
        print(f"Got exception when calling AWS Bedrock: {e}")
        traceback.print_exc()
        return "Error calling model"


async def main():

    prompts = ["tell me a joke", "count to 10"]
    tasks = [send_bedrock_request(prompt) for prompt in prompts]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)

asyncio.run(main())
