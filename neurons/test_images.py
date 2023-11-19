import requests
from openai import OpenAI

# Generate image with DALL-E 3
client = OpenAI()
response = client.images.generate(
    model="dall-e-3",
    prompt="a super cool image",
    size="1024x1024",
    quality="standard", # or hd (double the cost)
    style="vivid", # or natural
    n=1,
)

image_url = response.data[0].url
image_revised_prompt = response.data[0].revised_prompt
image_created = response.created
print(f"created at {image_created}\n\nrevised_prompt = {image_revised_prompt}\n\nurl = {image_url}")