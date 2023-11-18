# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import typing
import openai
import aiohttp
import bittensor as bt
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import requests
import re

def calculate_cosine_similarity(text1, text2):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity


# Give a perfect score as long as the miner's response is at least 90% similar to openai's response. Otherwise, give 0
def openai_score(openai_answer: str, response: str, weight: float) -> str:
    # stripped_openai = openai_answer.replace(" ", "").replace("\n", "").replace("\t", "")
    # stripped_response = response.replace(" ", "").replace("\n", "").replace("\t", "")

    similarity = calculate_cosine_similarity(openai_answer, response)
    bt.logging.debug(f"similarity is {similarity}")

    return weight if similarity > .75 else 0


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url_regex = (
    r'https://oaidalleapiprodscus\.blob\.core\.windows\.net/private/org-[\w-]+/'
    r'user-[\w-]+/img-[\w-]+\.(?:png|jpg)\?'
    r'st=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'(?:sp=\w+&)?'
    r'sv=\d{4}-\d{2}-\d{2}&'
    r'sr=\w+&'
    r'rscd=\w+&'
    r'rsct=\w+/[\w-]+&'
    r'skoid=[\w-]+&'
    r'sktid=[\w-]+&'
    r'skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'sks=\w+&'
    r'skv=\d{4}-\d{2}-\d{2}&'
    r'sig=[\w/%+=]+'
)

async def is_image_url(url):
    """Check if the URL points to an image asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                return response.status == 200 and 'image' in response.headers.get('Content-Type', '')
    except Exception as e:
        logging.error(f"Error checking URL: {e}")
        return False

async def load_image_from_url(url):
    """Load an image from a URL asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image.verify()  # Verify that this is indeed an image
                return image
    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        return None

def get_image_size(image):
    """Get the size of an image."""
    return image.size  # Returns a tuple (width, height)

def calculate_cosine_similarity(image, description, max_length=77):
    """Calculate the cosine similarity between a description and an image."""
    # ... existing logic ...

async def image_score(url, desired_size, description, weight, similarity_threshold=0.3):
    """Calculate the image score based on similarity and size asynchronously."""
    if not re.match(url_regex, url):
        logging.error("URL does not match the expected format.")
        return 0

    if not await is_image_url(url):
        logging.error("URL does not point to a valid image.")
        return 0

    image = await load_image_from_url(url)
    if image is None:
        logging.error("Failed to load image from URL.")
        return 0

    try:
        similarity = calculate_cosine_similarity(image, description)
        logging.debug(f"Similarity: {similarity}")

        size = get_image_size(image)
        logging.debug(f"Image size: {size}")

        if similarity > similarity_threshold and size == desired_size:
            return weight
        else:
            return 0
    except Exception as e:
        logging.error(f"Error in image scoring: {e}")
        return 0
