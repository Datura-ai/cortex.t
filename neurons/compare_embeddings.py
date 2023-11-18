import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import requests

def get_image_size(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(io.BytesIO(response.content))
        return image.size  # Returns a tuple (width, height)
    except requests.RequestException as e:
        raise Exception(f"Failed to load image: {e}")

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response was an unsuccessful status code
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except requests.RequestException as e:
        raise Exception(f"Failed to load image: {e}")

# Truncate the description to 77 tokens
description = "A close image portrait of an elegant Siamese cat. It has a white fur coat that shimmers under the mild sunlight. Noteworthy are its piercing blue almond-shaped eyes full of curiosity. The distinct color points on its ears, paws, and tail are a light cream shade. The cat's angular head, muscular body, and sleek short hair, classical characteristics of the Siamese breed, are all visible. The cat is sitting comfortably, looking attentively directly at the viewer, giving an overall impression of poised gracefulness., b64 = A close image portrait of an elegant Siamese cat. It has a white fur coat that shimmers under the mild sunlight. Noteworthy are its piercing blue almond-shaped eyes full of curiosity. The distinct color points on its ears, paws, and tail are a light cream shade. The cat's angular head, muscular body, and sleek short hair, classical characteristics of the Siamese breed, are all visible. The cat is sitting comfortably, looking attentively directly at the viewer, giving an overall impression of poised gracefulness."
max_length = 77  # Adjust as needed
inputs = processor(text=description, images=None, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

# Generate embeddings for the truncated description
text_embedding = model.get_text_features(**inputs)

# Generate embeddings for the image
image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-qCq4dk0xvMql4Wo8WT2B9QAv/user-ull71ynh86YyNbnhwT4G04KN/img-yMQXZJdLfmIXN4caB0KinMfO.png?st=2023-11-17T23%3A46%3A25Z&se=2023-11-18T01%3A46%3A25Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-17T20%3A06%3A25Z&ske=2023-11-18T20%3A06%3A25Z&sks=b&skv=2021-08-06&sig=W1Ajdn7GpJCp4WggsTbgYRgCDgnbZWacWrcyxBKaGzM%3D"
image = load_image_from_url(image_url)
inputs = processor(text=None, images=image, return_tensors="pt", padding=True, truncation=True)
image_embedding = model.get_image_features(**inputs)

# Calculate cosine similarity
similarity = torch.cosine_similarity(image_embedding, text_embedding, dim=1)

print(f"Cosine similarity: {similarity.item()}")
size = get_image_size(image_url)
print(f"size = {size}")
