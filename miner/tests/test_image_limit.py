import os
import asyncio
from openai import AsyncOpenAI


AsyncOpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not AsyncOpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

async def generate_image(prompt):
    client = AsyncOpenAI()

    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="standard",  # or hd (double the cost)
        style="vivid",  # or natural
        n=1,
    )

    return response

async def main():
    prompt = "a jeep ride through aruba on a majestical dirt sunny road in the forest"
    rate_to_test = 30
    while True:
        tasks = [generate_image(prompt) for _ in range(rate_to_test)]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            image_url = response.data[0].url
            image_revised_prompt = response.data[0].revised_prompt
            image_created = response.created
            print(f"created at {image_created}\n\nrevised_prompt = {image_revised_prompt}\n\nurl = {image_url}")

        await asyncio.sleep(60)  # Sleep for 60 seconds before next iteration

if __name__ == "__main__":
    asyncio.run(main())
