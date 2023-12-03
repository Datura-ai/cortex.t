import re
import ast
import regex
import traceback


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
        print(f"Error in preprocessing string: {traceback.format_exc()}")
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
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text)
        if match:
            list_str = match.group()

            evaluated = ast.literal_eval(list_str)
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        print(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    return text


text1 = """
```python entertainment_complex_questions = [     # Question 1     "Develop a comprehensive algorithm to predict the box office success of movies across different genres and global markets, considering variables such as cast popularity, marketing budget, release timing, and historical data of similar movies.",          # Question 2     "Create a neural network model that can generate original scripts for a TV series that match the linguistic style and thematic depth of a given showrunner, such as Aaron Sorkin or Shonda Rhimes, and test its effectiveness by comparing it with human-written scripts.",          # Question 3     "Design a virtual reality experience that integrates live performances with interactive audience participation, ensuring the experience is adaptable to multiple entertainment genres including concerts, theater, and sport events." ] ```
"""
text2 = """
 ```python economy_related_tasks = [     "Write a program that simulates the impact of a small tax cut on a simplified supply-demand model, considering only three industries, with relevance to broader economic factors being minimal.",     "Create a Monte Carlo simulation to project the potential variations in a local market's commodity prices, with little consideration for macroeconomic indicators such as GDP or employment rates.",     "Develop a script that scrapes data from a social media platform to gauge consumer sentiment on a new tech gadget, ignoring the broader economic implications of consumer electronics on the economy.",     "Implement an algorithm that predicts the stock price of a fictional company based on random fluctuations, without any connection to economic indicators like inflation or interest rates." ] ```
"""
text3 = """

"""

extract_python_list(text1)

