import re
import ast
import regex
import traceback


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
                    cleaned_text.append(char)  # Add the opening bracket
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


text1 = """
python entertainment_complex_questions = [     # Question 1     "Develop a comprehensive "algorithm" to predict the box office success of movies across different genres and global markets, considering variables such as cast popularity, marketing budget, release timing, and historical data of similar movies.",          # Question 2     "Create a neural network model that can generate original scripts for a TV series that match the linguistic style and thematic depth of a given showrunner, such as Aaron Sorkin or Shonda Rhimes, and test its effectiveness by comparing it with human-written scripts." ] 
"""

print(extract_python_list(text1))


