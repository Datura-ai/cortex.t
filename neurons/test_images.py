import requests
import wandb
from openai import OpenAI

# Initialize Weights & Biases
wandb.init(project="synthetic-images", entity="cortex-t")

# Generate image with DALL-E 3
client = OpenAI()
response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1,
)

# Get image URL
image_url = response.data[0].url

# Download the image
image_response = requests.get(image_url)
image_path = "images/image.jpg"
with open(image_path, "wb") as f:
    f.write(image_response.content)

# Log the image and prompt to wandb
wandb.log({"generated_image": wandb.Image(image_path), "prompt": "a white siamese cat"})

# Optional: finish the wandb run
wandb.finish()