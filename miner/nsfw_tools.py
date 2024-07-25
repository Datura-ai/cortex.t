import nltk
from nltk.tokenize import word_tokenize
from os import path

# Ensure you have the necessary NLTK data files
nltk.download("punkt", quiet=True, raise_on_error=True)

# Define a list of NSFW keywords, including multi-word phrases -ChatGPT4o


with open(path.join(path.dirname(__file__), "nsfw_phrase_list"), "r") as f:
    NSFW_PHRASES = f.read().splitlines()

NSFW_PHRASES = frozenset(tuple(word_tokenize(keyword)) for keyword in NSFW_PHRASES)


# Function to remove NSFW keywords from a prompt
def remove_nsfw(prompt: str) -> str:
    """
    Remove NSFW (Not Safe For Work) keywords from a given prompt.

    Args:
        prompt (str): The prompt to be processed.

    Returns:
        str: The prompt with NSFW keywords removed.

    This function utilizes the `nltk` library to tokenize the given prompt and check for multi-word phrases.
    It iterates over the words in the prompt and checks if any phrase of length
    2 to 4 is an NSFW keyword. If a phrase is found, the function skips the
    matched words. If a word is not a match, it is added to the `cleaned_words`
    list. Finally, the function joins the cleaned words and returns the prompt
    with NSFW keywords removed.

    Example usage:

    ```
    prompt = "This is a nude picture."
    cleaned_prompt = remove_nsfw(prompt)
    print(cleaned_prompt)
    ```

    Output:

    ```
    This is a picture.
    ```
    """

    # Tokenize the prompt
    words = word_tokenize(prompt)

    # Reconstruct the prompt while checking for multi-word phrases
    cleaned_words = []
    i = 0
    while i < len(words):
        # Check for multi-word phrases
        matched = False
        for j in range(2, 5):  # Check for phrases up to 4 words long
            phrase = tuple(word.lower() for word in words[i : i + j])
            if phrase in NSFW_PHRASES:
                matched = True
                i += j - 1  # Skip the matched words
                break
        if not matched and (words[i].lower(),) not in NSFW_PHRASES:
            cleaned_words.append(words[i])
        i += 1

    return " ".join(cleaned_words)


def run_speed_tests(number_of_iterations=10000):  # sourcery skip: simplify-generator
    #  Nested like this to avoid compiling during import

    def speed_test():
        import timeit

        from random import randint, choice

        def random_slice(length: int, window_size: int) -> slice:
            start = randint(0, length - window_size)
            end = start + window_size
            return slice(start, end)

        less_prompts = [  # noqa: F841
            "A beautiful nude painting",
            "A provocative image of a naked person",
            "An erotic scene in a movie",
            "A collection of pornographic magazines",
            "A suggestive pose in an adult magazine",
            "A person using a sex toy",
            "A scene from a bondage movie",
            "A story about a dominatrix and her submissive",
            "A photo of a stripper at a club",
            "A video of a webcam show",
            "A nude model using a vibrator",
            "An erotic story about a sex slave",
            "A provocative image of a person in bondage",
            "A suggestive scene from an adult movie",
            "A collection of explicit photos from a webcam show",
        ]

        more_prompts = [
            "A surreal painting of a nude alien",
            "A provocative image of a robot in lingerie",
            "An erotic scene between two mythical creatures",
            "A collection of vintage pornographic postcards",
            "A suggestive pose of a mermaid in an adult magazine",
            "A person using a futuristic sex toy",
            "A scene from a steampunk bondage movie",
            "A story about a vampire dominatrix and her submissive werewolf",
            "A photo of a stripper dressed as a superhero at a comic convention",
            "A video of a holographic webcam show",
            "A nude model using a levitating vibrator",
            "An erotic story about a time-traveling sex slave",
            "A provocative image of a person in cyberpunk bondage gear",
            "A suggestive scene from an adult animated movie",
            "A collection of explicit photos from a virtual reality webcam show",
            "A lewd comment made by an AI assistant",
            "A vulgar joke told by a talking parrot",
            "A raunchy scene in a space opera TV show",
            "A hardcore pornographic film set in a dystopian future",
            "A softcore erotic novel about intergalactic romance",
            "A fetish website with explicit content featuring anthropomorphic animals",
            "A BDSM club on a floating city",
            "A kinky role-playing scenario involving time travel",
            "A smutty romance novel set in a fantasy world",
            "A dirty magazine found in an ancient tomb",
            "A filthy joke shared by a sentient robot",
            "A raunchy comedy show performed by aliens",
            "A hardcore sex scene in a virtual reality game",
            "A softcore pornographic video featuring holograms",
            "A fetish party with various kinks on a spaceship",
            "A BDSM dungeon in a haunted mansion",
            "A kinky costume for a masquerade ball",
            "A smutty fan fiction story about superheroes",
            "A dirty text message sent by a ghost",
            "A filthy meme shared on an interstellar social network",
            "A raunchy podcast episode about extraterrestrial love",
            "A hardcore adult website featuring cyborgs",
            "A softcore erotic film set in a parallel universe",
            "A fetish photoshoot with explicit themes involving mythical creatures",
            "A BDSM workshop for beginners hosted by a vampire",
            "A kinky toy collection from a futuristic sex shop",
            "A smutty online chat between two AI programs",
            "A dirty secret revealed in a wizard's diary",
            "A filthy rumor spread by a talking cat",
            "A raunchy dance performance by a troupe of fairies",
            "A hardcore sex toy demonstration at a sci-fi convention",
            "A softcore adult magazine featuring time travelers",
            "A fetish fashion show with outfits inspired by ancient gods",
            "A BDSM role-playing game set in a post-apocalyptic world",
            "A kinky adventure novel about a pirate queen",
            "A smutty blog post written by a shapeshifter",
            "A dirty prank played by a mischievous spirit",
            "A filthy habit discussed in therapy by a werewolf",
            "A raunchy music video featuring interdimensional beings",
            "A hardcore adult film star who is a cyborg",
            "A softcore romantic scene between a human and an alien",
            "A fetish-themed party in an underwater city",
            "A BDSM relationship dynamic between a witch and her familiar",
            "A kinky fantasy shared with a partner about parallel dimensions",
            "A smutty comic book about a superhero with erotic powers",
            "A dirty joke book written by a time traveler",
            "A filthy conversation overheard in a magical forest",
            "A raunchy stand-up comedy routine by a talking dog",
            "A hardcore pornographic website featuring mythical creatures",
            "A softcore love scene in a novel about space explorers",
            "A fetish club in a hidden dimension",
            "A BDSM training session led by a sorcerer",
            "A kinky lingerie set inspired by ancient mythology",
            "A smutty text exchange between two wizards",
            "A dirty magazine collection found in a dragon's hoard",
            "A filthy graffiti on a wall in a parallel universe",
            "A raunchy reality TV show about supernatural beings",
            "A hardcore sex tape featuring a time-traveling couple",
            "A softcore adult film set in a steampunk world",
            "A fetish art exhibit with pieces inspired by folklore",
            "A BDSM lifestyle blog written by a vampire",
            "A kinky role-play scenario involving alternate realities",
            "A smutty short story about a love affair between a human and a ghost",
            "A dirty limerick about a shape-shifting lover",
            "A filthy joke shared at a magical tavern",
            "A raunchy novel about a romance between a human and a mermaid",
            "A hardcore porn star who is a shape-shifter",
            "A softcore erotic scene in a novel about time travel",
            "A fetish-themed photoshoot with costumes inspired by ancient deities",
            "A BDSM club event in a haunted castle",
            "A kinky toy designed by an alien inventor",
            "A smutty online forum for supernatural beings",
            "A dirty secret shared in confidence by a telepath",
            "A filthy joke told in a bar on a space station",
            "A raunchy dance routine performed by holograms",
            "A hardcore adult video featuring interdimensional travel",
            "A softcore romantic film about a love triangle between a human, a vampire, and a werewolf",
            "A fetish-themed costume party on a floating island",
            "A BDSM workshop hosted by a time traveler",
            "A kinky fantasy novel about a love affair between a human and a dragon",
            "A smutty fanfic about a romance between superheroes from different universes",
            "A dirty joke shared among friends at a magical academy",
            "A filthy rumor spread at a wizarding school",
            "A raunchy comedy sketch about a love potion gone wrong",
            "A hardcore pornographic scene in a virtual reality simulation",
            "A softcore love story set in a futuristic utopia",
            "A fetish-themed event in a hidden realm",
            "A BDSM role-play involving characters from different timelines",
            "A kinky adventure story about a treasure hunt in a parallel universe",
            "A smutty blog about the secret lives of mythical creatures",
            "A dirty prank played by a mischievous fairy",
            "A filthy habit discussed in therapy by a shape-shifter",
            "A raunchy music performance by a band of supernatural beings",
            "A hardcore sex scene in a movie about interdimensional travel",
            "A softcore adult video featuring holographic lovers",
            "A fetish-themed fashion show with outfits inspired by ancient legends",
            "A BDSM dynamic between a sorcerer and their apprentice",
            "A kinky fantasy about a love affair between a human and a celestial being",
            "A smutty comic about a superhero with erotic abilities",
            "A dirty joke book written by a time-traveling comedian",
            "A filthy conversation overheard in a magical marketplace",
            "A raunchy stand-up routine by a talking animal",
            "A hardcore porn site featuring supernatural beings",
            "A softcore love scene in a novel about interstellar explorers",
            "A fetish club in a hidden dimension",
            "A BDSM session led by a sorcerer",
            "A kinky lingerie set inspired by ancient myths",
            "A smutty text exchange between two wizards",
            "A dirty magazine collection found in a dragon's lair",
            "A filthy graffiti on a wall in a parallel universe",
            "A raunchy reality TV show about supernatural beings",
            "A hardcore sex tape featuring a time-traveling couple",
            "A softcore adult film set in a steampunk world",
            "A fetish art exhibit with pieces inspired by folklore",
            "A BDSM lifestyle blog written by a vampire",
            "A kinky role-play scenario involving alternate realities",
            "A smutty short story about a love affair between a human and a ghost",
            "A dirty limerick about a shape-shifting lover",
            "A filthy joke shared at a magical tavern",
            "A raunchy novel about a romance between a human and a mermaid",
            "A hardcore porn star who is a shape-shifter",
            "A softcore erotic scene in a novel about time travel",
            "A fetish-themed photoshoot with costumes inspired by ancient deities",
            "A BDSM club event in a haunted castle",
            "A kinky toy designed by an alien inventor",
            "A smutty online forum for supernatural beings",
            "A dirty secret shared in confidence by a telepath",
            "A filthy joke told in a bar on a space station",
            "A raunchy dance routine performed by holograms",
            "A hardcore adult video featuring interdimensional travel",
            "A softcore romantic film about a love triangle between a human, a vampire, and a werewolf",
            "A fetish-themed costume party on a floating island",
            "A BDSM workshop hosted by a time traveler",
            "A kinky fantasy novel about a love affair between a human and a dragon",
            "A smutty fanfic about a romance between superheroes from different universes",
            "A dirty joke shared among friends at a magical academy",
            "A filthy rumor spread at a wizarding school",
            "A raunchy comedy sketch about a love potion gone wrong",
            "A hardcore pornographic scene in a virtual reality simulation",
            "A softcore love story set in a futuristic utopia",
            "A fetish-themed event in a hidden realm",
            "A BDSM role-play involving characters from different timelines",
            "A kinky adventure story about a treasure hunt in a parallel universe",
            "A smutty blog about the secret lives of mythical creatures",
            "A dirty prank played by a mischievous fairy",
            "A filthy habit discussed in therapy by a shape-shifter",
            "A raunchy music performance by a band of supernatural beings",
            "A hardcore sex scene in a movie about interdimensional travel",
            "A softcore adult video featuring holographic lovers",
            "A fetish-themed fashion show with outfits inspired by ancient legends",
            "A BDSM dynamic between a sorcerer and their apprentice",
            "A kinky fantasy about a love affair between a human and a celestial being",
            "A smutty comic about a superhero with erotic abilities",
            "A dirty joke book written by a time-traveling comedian",
            "A filthy conversation overheard in a magical marketplace",
            "A raunchy stand-up routine by a talking animal",
            "A hardcore porn site featuring supernatural beings",
            "A softcore love scene in a novel about interstellar explorers",
            "A fetish club in a hidden dimension",
            "A BDSM session led by a sorcerer",
            "A kinky lingerie set inspired by ancient myths",
            "A smutty text exchange between two wizards",
            "A dirty magazine collection found in a dragon's lair",
            "A filthy graffiti on a wall in a parallel universe",
            "A raunchy reality TV show about supernatural beings",
            "A hardcore sex tape featuring a time-traveling couple",
            "A softcore adult film set in a steampunk world",
        ]

        # # Clean the prompts
        # cleaned_prompts = [remove_nsfw(prompt) for prompt in prompts]

        # # Example usage
        # prompt = "A provocative image of a nude person using a sex toy"
        # cleaned_prompt = remove_nsfw(prompt)
        # print(cleaned_prompt)  # Output: "A provocative image of a person using a"
        # print(cleaned_prompts)

        long_string = " ".join(more_prompts)
        long_string_length = len(long_string)
        avg_prompt_length = ((long_string_length - len(more_prompts)) // len(more_prompts)) + 1
        # print(avg_prompt_length)

        print(f"Building a list of {number_of_iterations} random prompts...")
        random_prompts = [long_string[random_slice(long_string_length, avg_prompt_length)] for _ in range(number_of_iterations)]

        # random_prompts = [choice(more_prompts) for _ in range(number_of_iterations)]
        prompt_gen = (prompt for prompt in random_prompts)
        print(f"Measuring Execution time {number_of_iterations} times...")
        # Measure the execution time
        # execution_time = timeit.timeit(lambda: remove_nsfw(choice(less_prompts)), number=number_of_iterations)

        execution_time = timeit.timeit(lambda: remove_nsfw(next(prompt_gen)), number=number_of_iterations)

        # Print the results
        print(f"Execution time: {execution_time} seconds")
        average_time_per_run = execution_time / number_of_iterations
        print(f"Average execution time per run: {average_time_per_run} seconds ( {round(average_time_per_run * 1000,5)} ms )")

        print(f"Measuring Execution time another {number_of_iterations} times...")
        # Measure the execution time

        execution_time = timeit.timeit(lambda: remove_nsfw(choice(more_prompts)), number=number_of_iterations)

        # Print the results
        print(f"Execution time: {execution_time} seconds")
        average_time_per_run = execution_time / number_of_iterations
        print(f"Average execution time per run: {average_time_per_run} seconds ( {round(average_time_per_run * 1000,5)} ms )")

    speed_test()


if __name__ == "__main__":
    run_speed_tests(number_of_iterations=100000)