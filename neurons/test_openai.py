import openai
import time
import os
import ast 
import traceback

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def send_openai_request(prompt, engine = "gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            stream=True
        )

        collected_messages = []
        for chunk in response:
            try:
                chunk_message = str(chunk['choices'][0]['delta']['content'])
            except:
                continue
            print(chunk_message)
            collected_messages.append(chunk_message)

        all_messages = ' '.join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling openai {e}")
        traceback.print_exc()  # This will print the full traceback
        return "Error calling model"

prompt = "count to 10"
print(send_openai_request(prompt))
