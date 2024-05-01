import requests

url = "https://api.elevenlabs.io/v1/text-to-speech/Rachel"

payload = {
    "text": "Text",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 123,
        "similarity_boost": 123,
        "style": 123,
        "use_speaker_boost": True
    },
    "pronunciation_dictionary_locators": [
        {
            "pronunciation_dictionary_id": "<string>",
            "version_id": "<string>"
        }
    ],
    "seed": 123
}
headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)