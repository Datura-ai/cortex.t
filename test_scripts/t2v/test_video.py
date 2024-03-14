import asyncio
import requests
import os
import io
import time
from openai import AsyncOpenAI
from playsound import playsound
import base64
import cv2 
from IPython.display import display, Image, Audio
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

api_key = os.getenv("STABILITY_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

AsyncOpenAI.api_key = openai_key
client = AsyncOpenAI(timeout=30)

async def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# stability video parameters 
# ref: https://platform.stability.ai/docs/api-reference#tag/v2alphageneration/paths/~1v2alpha~1generation~1image-to-video/post

# dimensions for video options: 
# 1024x576
# 576x1024
# 768x768

# seed = [ 0 .. 2147483648 ]
# default = 0
# description = A specific value that is used to guide the 'randomness' of the generation. (Omit this parameter or pass 0 to use a random seed.

# cfg_scale = [ 0 .. 10 ]
# default = 2.5
# description = How strongly the video sticks to the original image. Use lower values to allow the model more freedom to make changes and higher values to correct motion distortions.

# motion_bucket_id = [ 1 .. 255 ]
# default = 40
# description = Lower values generally result in less motion in the output video, while higher values generally result in more motion. This parameter corresponds to the motion_bucket_id parameter from here:
# https://static1.squarespace.com/static/6213c340453c3f502425776e/t/655ce779b9d47d342a93c890/1700587395994/stable_video_diffusion.pdf


# Step 1: generate an image with the appropriate parameters



# async def read_video_frames(video_path):
#     video = cv2.VideoCapture(video_path)
#     base64Frames = []
#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode(".jpg", frame)
#         base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
#     video.release()
#     return base64Frames

# async def display_frames(base64Frames):
#     display_handle = display(None, display_id=True)
#     for img in base64Frames:
#         display(Image(data=base64.b64decode(img.encode("utf-8"))))
#         await asyncio.sleep(0.025)

# async def generate_descriptions(base64Frames, model, max_tokens):
#     PROMPT_MESSAGES = [
#         {
#             "role": "user",
#             "content": [
#                 "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
#                 *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
#             ],
#         },
#     ]
#     params = {
#         "model": model,
#         "messages": PROMPT_MESSAGES,
#         "max_tokens": max_tokens,
#     }

#     result = await client.chat.completions.create(**params)
#     return result.choices[0].message.content

# async def generate_audio_speech(text):
#     response = requests.post(
#         "https://api.openai.com/v1/audio/speech",
#         headers={
#             "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
#         },
#         json={
#             "model": "tts-1-1106",
#             "input": text,
#             "voice": "onyx",
#         },
#     )

#     audio_file_path = "output_audio.mp3" 
#     with open(audio_file_path, "wb") as audio_file:
#         for chunk in response.iter_content(chunk_size=1024 * 1024):
#             audio_file.write(chunk)
#     return audio_file_path

# async def main():
#     image_path = "../t2i/image.jpg"
#     base64_image = await encode_image(image_path)

#     base64Frames = await read_video_frames("output.mp4")
#     print(len(base64Frames), "frames read.")

#     await display_frames(base64Frames)

#     description = await generate_descriptions(base64Frames, "gpt-4-vision-preview", 200)
#     print(description)

#     voiceover_script = await generate_descriptions(base64Frames, "gpt-4-vision-preview", 500)
#     print(voiceover_script)

#     audio_file_path = await generate_audio_speech(voiceover_script)
#     playsound(audio_file_path)


# asyncio.run(main())


# response = client.chat.completions.create(
#   model="gpt-4-vision-preview",
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What’s in this image?"},
#         {
#           "type": ,
#           "image_url": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#           },
#         },
#       ],
#     }
#   ],
#   max_tokens=300,
# )



image_path = "../images/john2.png"
seed = 0
cfg_scale = 1
motion_bucket_id = 150
FPS = 30 # 0 to 30



# response = requests.post(
#     "https://api.stability.ai/v2alpha/generation/image-to-video",
#     headers={
#         "authorization": api_key,
#     },
#     data={
#         "seed": seed,
#         "cfg_scale": cfg_scale,
#         "motion_bucket_id": motion_bucket_id,
#         "FPS": FPS,
#     },
#     files={
#         "image": ("file", open(image_path, "rb"), "image/png")
#     },
# )

# if response.status_code != 200:
#     raise Exception("Non-200 response: " + str(response.text))

# data = response.json()
# generation_id = data["id"]
# print(generation_id)
# time.sleep(30)


generation_id = "0ca93f4172448c1f0b5f40f3c5bb3afce3f60e8da4da07e9f635228be4e6acea"

response = requests.request(
    "GET",
    f"https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}",
    headers={
        'Accept': None, # Use 'application/json' to receive base64 encoded JSON
        'authorization': api_key,
    },
)

if response.status_code == 202:
    print("Generation in-progress, try again in 10 seconds.")
elif response.status_code == 200:
    print("Generation complete!")
    with open('./john5.mp4', 'wb') as file:
        file.write(response.content)
else:
    raise Exception("Non-200 response: " + str(response.json()))