import asyncio
import os
import traceback
from openai import OpenAI
from openai import AsyncOpenAI

os.environ["OPENAI_API_KEY"] = "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"
# OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
# if not OpenAI.api_key:
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(api_key="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", base_url="https://openrouter.ai/api/v1", timeout=60)


async def send_openai_request(prompt, engine="gpt-4-1106-preview"):
    try:
        messages = [{"role": "user", "content": "Explore the impact of Thomas Edison's inventions on modern society, such as the light bulb and phonograph"}]
        stream = await client.chat.completions.create(
            messages=messages,
            stream=True,
            model="openai/gpt-4o",
            seed=1234,
            temperature=0.0001,
            max_tokens=4096,
            # top_p=0.01,
            # top_k=1,
        )
        collected_messages = []

        async for part in stream:
            print(part.choices[0].delta.content or "")
            collected_messages.append(part.choices[0].delta.content or "")

        all_messages = "".join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling openai {e}")
        traceback.print_exc()
        return "Error calling model"


async def main():
    prompts = ["count to 10", "tell me a joke"]
    tasks = [send_openai_request(prompt) for prompt in prompts]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)


asyncio.run(main())
