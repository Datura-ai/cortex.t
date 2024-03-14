import base64
import time
import os
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

start_time = time.time()

# Set up the API connection
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0"
)

# Base64 Comparison
def base64_comparison(data1, data2):
    return data1 == data2

# Generate images and get their base64 representations
data = ('Stability', 'stable-diffusion-xl-1024-v1-0', 'halla halla, we dem boys', '1024x1024', 1024, 1024, 'standard', 'vivid', 656827, 30, None, 8.0, 'SAMPLER_K_DPMPP_2M', 9, 1)

meta = stability_api.generate(
    prompt=data[2],
    seed=data[8],
    steps=data[-6],
    cfg_scale=data[-4],
    width=data[-11],
    height=data[-10],
    samples=data[-1],
    sampler=data[-3],
)

# Convert image binary data to base64
base64_images = []
for image in meta:
    for artifact in image.artifacts:
        base64_images.append(base64.b64encode(artifact.binary).decode())
        base64_images.append(base64.b64encode(artifact.binary).decode())
    
# Time and compare using Base64 Comparison
base64_result = base64_comparison(base64_images[0], base64_images[1])
base64_duration = time.time() - start_time
print(f"Base64 Comparison: {base64_result}, Time taken: {base64_duration} seconds")

