

def preprocess_string(text):
    try:
        processed_text = text.replace("\t", " ")

        # Placeholder for single quotes within words
        placeholder = "___SINGLE_QUOTE___"

        # Replace single quotes within words with the placeholder
        processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, text)

        # Replace single quotes used for enclosing strings with double quotes
        processed_text = processed_text.replace("'", '"')

        # Restore the original single quotes from the placeholder
        processed_text = processed_text.replace(placeholder, "'")

        # Delete spaces after an opening bracket '['
        processed_text = re.sub(r"\[\s+", "[", processed_text)

        # Delete spaces before a closing bracket ']'
        processed_text = re.sub(r"\s+\]", "]", processed_text)

        # Remove characters before first '[' and after first ']'
        start = processed_text.find('[')
        end = processed_text.find(']')

        if start != -1 and end != -1 and end > start:
            processed_text = processed_text[start:end + 1]

        return processed_text

    except Exception as e:
        bt.logging.error(f"Error in preprocessing string: {traceback.format_exc()}")
        return text

def convert_to_list(text):
    pattern = r'\d+\.\s'
    items = [item.strip() for item in re.split(pattern, text) if item]
    return items


def extract_python_list(text: str):
    try:
        if re.match(r'\d+\.\s', text):
            return convert_to_list(text)
        bt.logging.info(f"Preprocessed text = {text}")
        text = preprocess_string(text)
        bt.logging.info(f"Postprocessed text = {text}")
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text)
        if match:
            list_str = match.group()

            evaluated = ast.literal_eval(list_str)
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        bt.logging.error(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    return text