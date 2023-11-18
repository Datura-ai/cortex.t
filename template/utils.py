import re
import ast
import json


def save_state_to_file(state, filename="state.json"):
    with open(filename, "w") as file:
        bt.logging.info(f"saved global state to {filename}")
        json.dump(state, file)

def load_state_from_file(filename="state.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            bt.logging.info("loaded previous state")
            return json.load(file)
    else:
        bt.logging.info("initialized new global state")
        return {
            "text": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0},
            "images": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0}
        }

def preprocess_string(text):
    try:
        # Placeholder for single quotes within words
        placeholder = "___SINGLE_QUOTE___"

        # Replace single quotes within words with the placeholder
        processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, text)

        # Replace single quotes used for enclosing strings with double quotes
        processed_text = processed_text.replace("'", '"')

        # Restore the original single quotes from the placeholder
        processed_text = processed_text.replace(placeholder, "'")

        return processed_text
    except Exception as e:
        logging.error(f"Error in preprocessing string: {e}")
        return text

def extract_python_list(text: str):
    try:
        text = preprocess_string(text)
        # Improved regex to match more complex list structures
        match = re.search(r'\[(?:[^\[\]]+|\[(?:[^\[\]]+|\[[^\[\]]*\])*\])*\]', text)
        if match:
            list_str = match.group()

            # Using ast.literal_eval to safely evaluate the string as a Python literal
            evaluated = ast.literal_eval(list_str)
            if isinstance(evaluated, list):
                return evaluated
        else:
            # Fallback mechanism if regex fails
            return fallback_list_extraction(text)

    except SyntaxError as e:
        bt.logging.error(f"Syntax error when extracting list: {e}\n{traceback.format_exc()}")
    except ValueError as e:
        bt.logging.error(f"Value error when extracting list: {e}\n{traceback.format_exc()}")
    except Exception as e:
        bt.logging.error(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    return None