import anthropic
import os


# Model options are claude-instant-1.2, claude-2.1
# Retrieve API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables")

client = anthropic.Anthropic()
client.api_key = api_key

question = "Tell me a short joke"
with client.beta.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": question}],
    model="claude-2.1",
) as stream:
  for text in stream.text_stream:
    print(text, end="", flush=True)

print("\n")


# non streaming, async
import asyncio
import traceback
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import bittensor as bt
import os


try:
    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except KeyError as exc:
    raise ValueError("Please set the ANTROPIC_API_KEY environment variable.") from exc

async def call_anthropic(messages, temperature, model, seed=1234) -> str:

    for _ in range(2):
        bt.logging.debug(f"Calling Anthropics. Model = {model}, Prompt = {prompt}")
        try:
            completion = anthropic.completions.create(
                model=model,
                max_tokens_to_sample=1000,
                prompt=f"{HUMAN_PROMPT} {messages[0]['content']}{AI_PROMPT}",
                temperature=temperature,
            )
            response = completion.completion
            bt.logging.debug(f"Validator response is {response}")
            return response

        except Exception as e:
            bt.logging.error(f"Error when calling Anthropics: {traceback.format_exc()}")
            await asyncio.sleep(0.5)

    return None

# Example usage of the function
prompt = "Tell me a short joke"
messages = [{'role': 'user', 'content': prompt}]
model = "claude-2"
temperature = 0.0001

# Run the async function
response = asyncio.run(call_anthropic(messages, temperature, model))
print(response)