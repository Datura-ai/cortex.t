import os
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

google_api = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=google_api)

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is the meaning of life?", stream=True)


# from PIL import Image

# img = Image.open('image.jpg')
# # img.show()

# # Initialize and use the model
# model = genai.GenerativeModel('gemini-pro-vision')
# response = model.generate_content(img)

# print(response.text)
