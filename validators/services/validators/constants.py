TEXT_MODEL = "gpt-4-turbo-2024-04-09"
TEXT_PROVIDER = "OpenAI"
TEXT_MAX_TOKENS = 4096
TEXT_TEMPERATURE = 0.001
TEXT_WEIGHT = 1
TEXT_SEED = 1234
TEXT_TOP_P = 0.01
TEXT_TOP_K = 1
VISION_MODELS = ["gpt-4o", "claude-3-opus-20240229", "anthropic.claude-3-sonnet-20240229-v1:0",
                 "claude-3-5-sonnet-20240620"]
TEXT_VALI_MODELS_WEIGHTS = {
    "AnthropicBedrock": {
        "anthropic.claude-v2:1": 1
    },
    "OpenAI": {
        "gpt-4o": 1,
        "gpt-4-1106-preview": 1,
        "gpt-3.5-turbo": 1000,
        "gpt-3.5-turbo-16k": 1,
        "gpt-3.5-turbo-0125": 1,
    },
    "Gemini": {
        "gemini-pro": 1,
        "gemini-1.5-flash": 1,
        "gemini-1.5-pro": 1,
    },
    "Anthropic": {
        "claude-3-5-sonnet-20240620": 1,
        "claude-3-opus-20240229": 1,
        "claude-3-sonnet-20240229": 1,
        "claude-3-haiku-20240307": 1000,
    },
    "Groq": {
        "gemma-7b-it": 1000,
        "llama3-70b-8192": 1,
        "llama3-8b-8192": 1,
        "mixtral-8x7b-32768": 1,
    },
    "Bedrock": {
        # "anthropic.claude-3-sonnet-20240229-v1:0": 1,
        "cohere.command-r-v1:0": 1,
        # "meta.llama2-70b-chat-v1": 1,
        # "amazon.titan-text-express-v1": 1,
        "mistral.mistral-7b-instruct-v0:2": 1,
        "ai21.j2-mid-v1": 1,
        # "anthropic.claude-3-5-sonnet-20240620-v1:0": 1,
        # "anthropic.claude-3-opus-20240229-v1:0": 1,
        # "anthropic.claude-3-haiku-20240307-v1:0": 1
    }
}
