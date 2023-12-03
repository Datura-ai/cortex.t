import re
import ast
import regex
import traceback

def preprocess_string(text):
    processed_text = text.replace("\t", "")
    placeholder = "___SINGLE_QUOTE___"
    processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, processed_text)
    processed_text = processed_text.replace("'", '"').replace(placeholder, "'")

    # First, remove all comments, ending at the next quote
    no_comments_text = ""
    i = 0
    in_comment = False
    while i < len(processed_text):
        if processed_text[i] == '#':
            in_comment = True
        elif processed_text[i] == '"' and in_comment:
            in_comment = False
            no_comments_text += processed_text[i]  # Keep the quote that ends the comment
            i += 1
            continue
        if not in_comment:
            no_comments_text += processed_text[i]
        i += 1

    # Now process the text without comments for quotes
    cleaned_text = []
    inside_quotes = False
    found_first_bracket = False

    i = 0
    while i < len(no_comments_text):
        char = no_comments_text[i]

        if not found_first_bracket:
            if char == '[':
                found_first_bracket = True
            cleaned_text.append(char)
            i += 1
            continue

        if char == '"':
            # Look for preceding comma or bracket, skipping spaces
            preceding_char_index = i - 1
            found_comma_or_bracket = False

            while preceding_char_index >= 0:
                if no_comments_text[preceding_char_index] in '[,':  # Check for comma or opening bracket
                    found_comma_or_bracket = True
                    break
                elif no_comments_text[preceding_char_index] not in ' \n':  # Ignore spaces and new lines
                    break
                preceding_char_index -= 1

            following_char_index = i + 1
            while following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in ' \n':
                following_char_index += 1

            if found_comma_or_bracket or \
               (following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in '],'):
                inside_quotes = not inside_quotes
            else:
                i += 1
                continue  # Skip this quote

            cleaned_text.append(char)
            i += 1
            continue

        if char == ' ':
            # Skip spaces if not inside quotes and if the space is not between words
            if not inside_quotes and (i == 0 or no_comments_text[i - 1] in ' ,[' or no_comments_text[i + 1] in ' ,]'):
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


