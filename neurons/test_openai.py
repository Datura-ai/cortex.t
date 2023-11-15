import asyncio
import os
import traceback
from openai import OpenAI
from openai import AsyncOpenAI

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI()

async def send_openai_request(prompt, engine="gpt-4-1106-preview"):
    try:
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            model=engine,
        )
        collected_messages = []

        async for part in stream:
            print(part.choices[0].delta.content or "")
            collected_messages.append(part.choices[0].delta.content or "")

        all_messages = ''.join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling openai {e}")
        traceback.print_exc()  # This will print the full traceback
        return "Error calling model"

async def main():
    prompts = ["count to 10", "tell me a joke"]
    tasks = [send_openai_request(prompt) for prompt in prompts]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)

asyncio.run(main())
