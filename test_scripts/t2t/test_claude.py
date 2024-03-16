from anthropic import AsyncAnthropic
import os
import asyncio
# https://docs.anthropic.com/claude/reference/messages_post
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables")

claude_client = AsyncAnthropic()
claude_client.api_key = api_key

messages = [
    {
        "role": "system",
        "content": "respond in spanish"
    },
    {
        "role": "user",
        "content": "Hello!"
    }
]
max_tokens = 100
model = "claude-3-opus-20240229"


# streaming
async def call_claude(messages, max_tokens, model):
    system_prompt = None
    filtered_messages = []
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
        else:
            filtered_messages.append(message)

    stream_kwargs = {
        "max_tokens": max_tokens,
        "messages": filtered_messages,
        "model": model,
    }

    if system_prompt:
        stream_kwargs["system"] = system_prompt

    completion = claude_client.messages.stream(**stream_kwargs)
    async with completion as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    # Send final message to close the stream
    print("\n")


# non streaming
# async def call_claude(messages, max_tokens, model):
#     filtered_messages = []
#     for message in messages:
#         if message["role"] == "system":
#             system_prompt = message["content"]
#         else:
#             filtered_messages.append(message)

#     kwargs = {
#         "max_tokens": max_tokens,
#         "messages": filtered_messages,
#         "model": model,
#     }

#     if system_prompt:
#         kwargs["system"] = system_prompt
       
#     message = await claude_client.messages.create(**kwargs)
#     print(message.content[0].text)
#     return message.content[0].text

async def main():
    await call_claude(messages, max_tokens, model)

if __name__ == "__main__":
    asyncio.run(main())
