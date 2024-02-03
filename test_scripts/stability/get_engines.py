import os
import requests

# image model list
# ['esrgan-v1-x2plus', 'stable-diffusion-xl-1024-v0-9', 'stable-diffusion-xl-1024-v1-0', 'stable-diffusion-v1-6', 'stable-diffusion-512-v2-1', 'stable-diffusion-xl-beta-v2-2-2']

api_host = os.getenv('API_HOST', 'https://api.stability.ai')
url = f"{api_host}/v1/engines/list"

api_key = os.getenv("STABILITY_API_KEY")
if api_key is None:
    raise Exception("Missing Stability API key.")

response = requests.get(url, headers={
    "Authorization": f"Bearer {api_key}"
})

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

# Do something with the payload...
payload = response.json()
print(payload)
engine_ids = [engine['id'] for engine in payload]
print(engine_ids)