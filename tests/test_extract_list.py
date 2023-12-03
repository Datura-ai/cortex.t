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
                cleaned_text.append(char)
                i += 1
                continue

            if char == '#':
                while i < len(processed_text) and processed_text[i] not in ['\n', '"']:
                    i += 1
                continue

            if char == '"':
                # Check for preceding or following bracket or comma, ignoring spaces and comments
                preceding_char_index = i - 1
                while preceding_char_index > 0 and processed_text[preceding_char_index] in ' \n':
                    preceding_char_index -= 1

                following_char_index = i + 1
                while following_char_index < len(processed_text) and processed_text[following_char_index] in ' \n':
                    following_char_index += 1

                if (preceding_char_index >= 0 and processed_text[preceding_char_index] in '[,') or \
                   (following_char_index < len(processed_text) and processed_text[following_char_index] in '],'):
                    inside_quotes = not inside_quotes
                else:
                    i += 1
                    continue  # Skip this quote

                cleaned_text.append(char)
                i += 1
                continue

            if char == ' ':
                # Skip spaces if not inside quotes and if the space is not between words
                if not inside_quotes and (i == 0 or processed_text[i - 1] in ' ,[' or processed_text[i + 1] in ' ,]'):
                    i += 1
                    continue

            cleaned_text.append(char)
            i += 1

        cleaned_str = ''.join(cleaned_text)
        cleaned_str = re.sub(r"\[\s+", "[", cleaned_str)
        cleaned_str = re.sub(r"\s+\]", "]", cleaned_str)
        cleaned_str = re.sub(r"\s*,\s*", ", ", cleaned_str)  # Ensure single space after commas

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
my_list = [     # Question 1     "Develop a comprehensive "algorithm" to make predictions, considering variables such as cast 'popularity' and marketing budget.",          # Question 2     "Create a neural network model that can generate ""original"" scripts for a TV series" ] 
"""

print(extract_python_list(text1))


