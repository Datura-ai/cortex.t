import anthropic
import os

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