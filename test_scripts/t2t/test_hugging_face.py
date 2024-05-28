import asyncio
import os
import traceback
from huggingface_hub import AsyncInferenceClient

api_key = os.environ.get("HUGGING_FACE_API_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables")

hugging_face_client = AsyncInferenceClient(token=api_key)

async def send_hugging_face_request(prompt, model="HuggingFaceH4/zephyr-7b-beta"):
    try:
        stream = await hugging_face_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            model=model,
            seed=1233,
            temperature=0.001,
            max_tokens=1,
        )
        collected_messages = []
        async for token in stream:
            print(token.choices[0].delta.content or "")
            collected_messages.append(token.choices[0].delta.content or "")

        all_messages = ''.join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling Hugging Face {e}")
        traceback.print_exc()
        return "Error calling model"


async def main():
    prompts = ["count to 10", "tell me a joke"]
    tasks = [send_hugging_face_request(prompt) for prompt in prompts]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)

asyncio.run(main())
