import os
import io
import requests
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

engine_id = "stable-diffusion-v1-6"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = os.getenv("STABILITY_API_KEY")


# Set up the API connection
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_API_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0"
)

# https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage


# engine_ids
# ['esrgan-v1-x2plus', 'stable-diffusion-xl-1024-v0-9', 'stable-diffusion-xl-1024-v1-0', 'stable-diffusion-v1-6', 'stable-diffusion-512-v2-1', 'stable-diffusion-xl-beta-v2-2-2']

# height
# multiple of 64 >= 128
# default: 512 or 1024 depending on the model

# width
# multiple of 64 >= 128
# default: 512 or 1024 depending on the model

# Engine-specific dimension validation:
# SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
# SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
# SDXL v1.0: same as SDXL v0.9
# SD v1.6: must be between 320x320 and 1536x1536

# text_prompts
# a list of prompts to use for generation

# weight
# positive to prompt the image, negative to prompt the image away from the text
# total possible range is [-10, 10] but we recommend staying within the range of [-2, 2].
# example:
# prompt= [generation.Prompt(text="beautiful night sky above japanese town, anime style",parameters=generation.PromptParameters(weight=1)),
    # generation.Prompt(text="clouds",parameters=generation.PromptParameters(weight=-1))],
# this will not have clouds in the output

# cfg_scale
# [ 0 .. 35 ]
# default = 7
# description: How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)

# clip_guidance_preset	
# FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST
# default = NONE


# seed
# [ 0 .. 4294967295 ]
# default = 0
# If a seed is provided, the resulting generated image will be deterministic.


# samplers
# ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde
# default = will auto pick an appropriate one

# samples
# [ 1 .. 10 ]
# default = 1
# number of images to generate


# steps
# [ 10 .. 50 ]
# default = 30
# Number of diffusion steps to run.

# style_preset
# 3d-model analog-film anime cinematic comic-book digital-art enhance fantasy-art isometric line-art low-poly modeling-compound neon-punk origami photographic pixel-art tile-texture
# Pass in a style preset to guide the image model towards a particular style. 


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

prompt1 = 'a dog running in a park'
weight1 = 1.5
prompt2 = 'grass'
weight2 = -1

# data = ('Stability', 'stable-diffusion-512-v2-1', prompt, '576x1024', 576, 1024, 'standard', 'vivid', 656827, 30, None, 8.0, 'SAMPLER_K_DPMPP_2M', 9, 1)

meta = stability_api.generate(
    prompt=[generation.Prompt(text=prompt1,parameters=generation.PromptParameters(weight=weight1)),
    generation.Prompt(text=prompt2,parameters=generation.PromptParameters(weight=weight2))],
    seed=100,
    steps=50,
    cfg_scale=17,
    width=576,
    height=1024,
    samples=5,
)

# Process and upload the image
import matplotlib.pyplot as plt
import numpy as np

for image in meta:
    for artifact in image.artifacts:
        img_array = np.frombuffer(artifact.binary, dtype=np.uint8)
        img = plt.imread(io.BytesIO(img_array), format='PNG')
        plt.imshow(img)
        plt.axis('off')  # Do not show axis to mimic the original display
        plt.show()
        response = requests.post('https://file.io', files={'file': io.BytesIO(img_array)})
        if response.status_code == 200:
            image_url = response.json()['link']
            print(image_url)
        else:
            bt.logging.error("Failed to upload the image.")