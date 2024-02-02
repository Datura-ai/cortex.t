import os
import io
import requests
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Set up the API connection
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0"
)

# # Generate the image
# answers = stability_api.generate(
#     prompt="expansive landscape rolling greens with gargantuan yggdrasil, intricate world-spanning roots towering under a blue alien sky, masterful, ghibli",
#     seed=9283409,
#     steps=30,
#     cfg_scale=8.0,
#     width=1024,
#     height=1024,
#     samples=1,
#     sampler=generation.SAMPLER_K_DPMPP_2M
# )

# # Process and upload the image
# for resp in answers:
#     for artifact in resp.artifacts:
#         if artifact.finish_reason == generation.FILTER:
#             print("Safety filters activated, prompt could not be processed.")
#         elif artifact.type == generation.ARTIFACT_IMAGE:
#             img = Image.open(io.BytesIO(artifact.binary))
#             img_buffer = io.BytesIO()
#             img.save(img_buffer, format="PNG")
#             img_buffer.seek(0)

#             # Upload to file.io
#             response = requests.post('https://file.io', files={'file': img_buffer})
#             if response.status_code == 200:
#                 print(f"Image uploaded successfully. URL: {response.json()['link']}")
#             else:
#                 print("Failed to upload the image.")

prompt = 'halla halla, we dem boys'

data = ('Stability', 'stable-diffusion-xl-1024-v1-0', prompt, '1024x1024', 1024, 1024, 'standard', 'vivid', 656827, 30, None, 8.0, 'SAMPLER_K_DPMPP_2M', 9, 1)

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

# Process and upload the image
for image in meta:
    for artifact in image.artifacts:
        img = Image.open(io.BytesIO(artifact.binary))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        # Upload to file.io
        response = requests.post('https://file.io', files={'file': img_buffer})
        if response.status_code == 200:
            image_url = response.json()['link']
            print(image_url)
        else:
            bt.logging.error("Failed to upload the image.")