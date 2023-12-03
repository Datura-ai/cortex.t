import re
import ast
import regex
import traceback

import re

def preprocess_string(text):
    try:
        processed_text = text.replace("\t", "")
        placeholder = "___SINGLE_QUOTE___"
        processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, processed_text)
        processed_text = processed_text.replace("'", '"').replace(placeholder, "'")

        cleaned_text = []
        inside_quotes = False
        found_first_bracket = False

        i = 0
        while i < len(processed_text):
            char = processed_text[i]

            if not found_first_bracket:
                if char == '[':
                    found_first_bracket = True
                i += 1
                continue

            if char == '#':
                # Skip until the end of the line or the string
                while i < len(processed_text) and processed_text[i] not in ['\n', '"']:
                    i += 1
                continue

            if char == '"':
                inside_quotes = not inside_quotes

            if not inside_quotes and char == ' ' and (i == 0 or processed_text[i - 1] == ' '):
                i += 1
                continue

            cleaned_text.append(char)
            i += 1

        cleaned_str = ''.join(cleaned_text)

        # Clean up bracket spacing
        cleaned_str = re.sub(r"\[\s+", "[", cleaned_str)
        cleaned_str = re.sub(r"\s+\]", "]", cleaned_str)

        # Extract the portion within the outermost square brackets
        start, end = cleaned_str.find('['), cleaned_str.rfind(']')
        if start != -1 and end != -1 and end > start:
            cleaned_str = cleaned_str[start:end + 1]

        return cleaned_str
    except Exception as e:
        print(f"Error in preprocessing string: {e}")
        return text



def convert_to_list(text):
    pattern = r'\d+\.\s'
    items = [item.strip() for item in re.split(pattern, text) if item]
    return items

def extract_python_list(text: str):
    try:
        if re.match(r'\d+\.\s', text):
            return convert_to_list(text)
        
        print(f"Preprocessed text = {text}")
        text = preprocess_string(text)
        print(f"Postprocessed text = {text}")

        # Extracting list enclosed in square brackets
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text, re.DOTALL)
        if match:
            list_str = match.group(1)

            # Using ast.literal_eval to safely evaluate the string as a list
            evaluated = ast.literal_eval('[' + list_str + ']')
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        print(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    return text


text2 = """
    ["List five synonyms for "write".",     "Name two famous authors from the 20th century.",     "Write a sentence using the word "pen" as a metaphor.",     "Describe your favorite book in three words.",     "What is the primary tool you use for writing? List one and explain why.",     "Identify and write down one goal for your writing this month.",     "List three adjectives to describe your current writing project or interest."]
 """

text3 = """
["An artist's cluttered studio, filled with whispered critiques and the echoes of brushstrokes on canvas.",     "A haunting battlefield from World War I, with the whispers of soldiers" prayers echoing through the abandoned trenches.",     "A virtual reality world where every user's whispers echo through digital landscapes, creating an immersive soundscape.",     "A Victorian-era seance room, with whispered messages from the beyond echoing around a dimly lit, ornate table.",     "A cavernous wine cellar, where the whispered tales of harvests past echo among the dusty, age-old bottles.",     "A supernatural vortex in the heart of a swirling nebula, where the whispers of the universe echo the origins of time."]
"""

# this passes
text1 = """
python entertainment_complex_questions = [     # Question 1     "Develop a comprehensive algorithm to predict the box office success of movies across different genres and global markets, considering variables such as cast popularity, marketing budget, release timing, and historical data of similar movies.",          # Question 2     "Create a neural network model that can generate original scripts for a TV series that match the linguistic style and thematic depth of a given showrunner, such as Aaron Sorkin or Shonda Rhimes, and test its effectiveness by comparing it with human-written scripts.",          # Question 3     "Design a virtual reality experience that integrates live performances with interactive audience participation, ensuring the experience is adaptable to multiple entertainment genres including concerts, theater, and sport events." ] 
"""

print(extract_python_list(text1))

