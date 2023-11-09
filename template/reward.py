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

import typing
import openai
import bittensor as bt
import difflib

def compare_texts(openai, response):
    # Tokenize the texts into words
    openai = openai.split()
    response = response.split()

    # Initialize the SequenceMatcher
    matcher = difflib.SequenceMatcher(None, openai, response)

    # Get ratio of similarity considering the order
    similarity_ratio = matcher.ratio()

    return similarity_ratio

# Give a perfect score as long as the miner's response is at least 90% similar to openai's response. Otherwise, give 0
def openai_score(openai_answer: str, response: str) -> str:
    bt.logging.info(f"openai_answer = {openai_answer}")
    bt.logging.info(f"response = {response}\n")
    stripped_openai = openai_answer.replace(" ", "").replace("\n", "").replace("\t", "")
    stripped_response = response.replace(" ", "").replace("\n", "").replace("\t", "")

    similarity = compare_texts(stripped_openai, stripped_response)
    bt.logging.info(f"similarity is {similarity}")

    return 1.0 if similarity > .90 else 0

    
