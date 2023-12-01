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

__version__ = "1.2.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

import os
from openai import AsyncOpenAI
AsyncOpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not AsyncOpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=30.0)

# Blacklist variables
ALLOW_NON_REGISTERED = False
PROMPT_BLACKLIST_STAKE = 50
IMAGE_BLACKLIST_STAKE = 50
EMBEDDING_BLACKLIST_STAKE = 50
ISALIVE_BLACKLIST_STAKE = min(PROMPT_BLACKLIST_STAKE, IMAGE_BLACKLIST_STAKE)
MIN_REQUEST_PERIOD = 2
MAX_REQUESTS = 40
# must have the test_key whitelisted to avoid a global blacklist
test_key = "5DcRHcCwD33YsHfj4PX5j2evWLniR1wSWeNmpf5RXaspQT6t"
corcel = "5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8"

weight_copiers = []
WHITELISTED_KEYS = [test_key, corcel]
running_wandb = [149, 255, 57, 109, 81, 124, 16, 26, 178, 103, 146, 0, 1, 244, 113, 42]
actual_validator_uids = [244, 0, 103, 81, 117, 178, 1, 26, 224, 214, 109, 16, 104, 2, 124, 102, 7, 135, 114, 251, 113, 4]
validators_running_wandb = [0, 1, 16, 26, 81, 103, 109, 113, 124, 178, 244]
BLACKLISTED_KEYS = []


question_themes = ['Love and relationships', 'Nature and environment', 'Art and creativity', 'Technology and innovation', 'Health and wellness', 'History and culture', 'Science and discovery', 'Philosophy and ethics', 'Education and learning', 'Music and rhythm', 'Sports and athleticism', 'Food and nutrition', 'Travel and adventure', 'Fashion and style', 'Books and literature', 'Movies and entertainment', 'Politics and governance', 'Business and entrepreneurship', 'Mind and consciousness', 'Family and parenting', 'Social media and networking', 'Religion and spirituality', 'Money and finance', 'Language and communication', 'Human behavior and psychology', 'Space and astronomy', 'Climate change and sustainability', 'Dreams and aspirations', 'Equality and social justice', 'Gaming and virtual reality', 'Artificial intelligence and robotics', 'Creativity and imagination', 'Emotions and feelings', 'Healthcare and medicine', 'Sportsmanship and teamwork', 'Cuisine and gastronomy', 'Historical events and figures', 'Scientific advancements', 'Ethical dilemmas and decision making', 'Learning and growth', 'Music genres and artists', 'Film genres and directors', 'Government policies and laws', 'Startups and innovation', 'Consciousness and perception', 'Parenting styles and techniques', 'Online communities and forums', 'Religious practices and rituals', 'Personal finance and budgeting', 'Linguistic diversity and evolution', 'Human cognition and memory', 'Astrology and horoscopes', 'Environmental conservation', 'Personal development and self-improvement', 'Sports strategies and tactics', 'Culinary traditions and customs', 'Ancient civilizations and empires', 'Medical breakthroughs and treatments', 'Moral values and principles', 'Critical thinking and problem solving', 'Musical instruments and techniques', 'Film production and cinematography', 'International relations and diplomacy', 'Corporate culture and work-life balance', 'Neuroscience and brain function', 'Childhood development and milestones', 'Online privacy and cybersecurity', 'Religious tolerance and understanding', 'Investment strategies and tips', 'Language acquisition and fluency', 'Social influence and conformity', 'Space exploration and colonization', 'Sustainable living and eco-friendly practices', 'Self-reflection and introspection', 'Sports psychology and mental training', 'Globalization and cultural exchange', 'Political ideologies and systems', 'Entrepreneurial mindset and success', 'Conscious living and mindfulness', 'Positive psychology and happiness', 'Music therapy and healing', 'Film analysis and interpretation', 'Human rights and advocacy', 'Financial literacy and money management', 'Multilingualism and translation', 'Social media impact on society', 'Religious extremism and radicalization', 'Real estate investment and trends', 'Language preservation and revitalization', 'Social inequality and discrimination', 'Climate change mitigation strategies', 'Self-care and well-being', 'Sports injuries and rehabilitation', 'Artificial intelligence ethics', 'Creativity in problem solving', 'Emotional intelligence and empathy', 'Healthcare access and affordability', 'Sports analytics and data science', 'Cultural appropriation and appreciation', 'Ethical implications of technology']
text_questions = ['What is the most important quality you look for in a partner?', 'How do you define love?', 'What is the most romantic gesture you have ever received?', 'What is your favorite love song and why?', 'What is the key to a successful long-term relationship?', 'What is your idea of a perfect date?', 'What is the best piece of relationship advice you have ever received?', 'What is the most memorable love story you have heard?', 'What is the biggest challenge in maintaining a healthy relationship?', 'What is your favorite way to show someone you love them?']

image_themes = ["Urban Echoes", "Nature's Whispers", "Futuristic Visions", "Emotional Abstracts", "Memory Fragments", "Mythical Echoes", "Underwater Mysteries", "Cosmic Wonders", "Ancient Secrets", "Cultural Tapestries", "Wild Motion", "Dreamlike States", "Seasonal Shifts", "Nature's Canvas", "Night Lights", "Historical Shadows", "Miniature Worlds", "Desert Dreams", "Robotic Integrations", "Fairy Enchantments", "Timeless Moments", "Dystopian Echoes", "Animal Perspectives", "Urban Canvas", "Enchanted Realms", "Retro Futures", "Emotive Rhythms", "Human Mosaics", "Undersea Unknowns", "Mystical Peaks", "Folklore Reimagined", "Outer Realms", "Vintage Styles", "Urban Wilderness", "Mythical Retellings", "Colorful Breezes", "Forgotten Places", "Festive Illuminations", "Masked Realities", "Oceanic Legends", "Digital Detachments", "Past Reverberations", "Shadow Dances", "Future Glimpses", "Wild Forces", "Steampunk Realms", "Reflective Journeys", "Aerial Grace", "Microscopic Worlds", "Forest Spirits"]
image_questions = ['A majestic golden eagle soaring high above a mountain range, its powerful wings spread wide against a clear blue sky.', 'A bustling medieval marketplace, full of colorful stalls, various goods, and people dressed in period attire, with a castle in the background.', 'An underwater scene showcasing a vibrant coral reef teeming with diverse marine life, including fish, sea turtles, and starfish.', 'A serene Zen garden with neatly raked sand, smooth stones, and a small, gently babbling brook surrounded by lush green foliage.', 'A futuristic cityscape at night, illuminated by neon lights, with flying cars zooming between towering skyscrapers.', 'A cozy cabin in a snowy forest at twilight, with warm light glowing from the windows and smoke rising from the chimney.', 'A surreal landscape with floating islands, cascading waterfalls, and a path leading to a castle in the sky, set against a sunset backdrop.', 'An astronaut exploring the surface of Mars, with a detailed spacesuit, the red Martian terrain around, and Earth visible in the sky.', 'A lively carnival scene with a Ferris wheel, colorful tents, crowds of happy people, and the air filled with the smell of popcorn and cotton candy.', 'A majestic lion resting on a savanna, with the African sunset in the background, highlighting its powerful mane and serene expression.']


# Import all submodules.
from . import protocol
from . import reward
from . import utils
