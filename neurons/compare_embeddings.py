import torch
from PIL import Image
import requests
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.nn.functional import cosine_similarity
import clip

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(requests.get(image_path, stream=True).raw) if image_path.startswith('http') else Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

# Generate embedding for the image
image_embedding = model.encode_image(preprocess_image("https://oaidalleapiprodscus.blob.core.windows.net/private/org-qCq4dk0xvMql4Wo8WT2B9QAv/user-ull71ynh86YyNbnhwT4G04KN/img-j0JOv1m3nKl46pvJrYB8QDEw.png?st=2023-11-17T20%3A54%3A11Z&se=2023-11-17T22%3A54%3A11Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-17T19%3A56%3A59Z&ske=2023-11-18T19%3A56%3A59Z&sks=b&skv=2021-08-06&sig=zCgyt2uiqhc/qOUNUkA%2BIYKwvowzY/Zb0f%2B11a2Qh7I%3D"))

# Generate embedding for the text
text = clip.tokenize(["A stately lion, full of regality and power, lying at rest in the midst of an expansive savanna. The scene is illuminated by a striking African sunset, casting a warm and magical glow. This light captures each strand of the lion's impressive mane, further intensifying its majesty. At the same time, it also softens the lion's expression, which carries a serene and peaceful aura. This breathtaking landscape seems to exist in perfect harmony with and enhances the grandeur of the lion, the undisputed king of this realm."]).to(device)
text_embedding = model.encode_text(text)

# Calculate cosine similarity
similarity = cosine_similarity(image_embedding, text_embedding)

print(f"Cosine similarity: {similarity.item()}")
