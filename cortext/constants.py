TEXT_MODEL = "gpt-4-turbo-2024-04-09"
TEXT_PROVIDER = "OpenAI"
TEXT_MAX_TOKENS = 4096
TEXT_TEMPERATURE = 0.001
TEXT_WEIGHT = 1
TEXT_TOP_P = 0.01
TEXT_TOP_K = 1
VISION_MODELS = ["gpt-4o", "claude-3-opus-20240229", "anthropic.claude-3-sonnet-20240229-v1:0",
                 "claude-3-5-sonnet-20240620"]
TEXT_VALI_MODELS_WEIGHTS = {
    # from https://openai.com/api/pricing/
    "OpenAI": {
        "gpt-4o": 15.00,
        # "gpt-4o-mini": 0.600,
        # "gpt-3.5-turbo": 2.00,
        # "o1-preview": 60.00,
        # "o1-mini": 12.00,
    },
    # from https://ai.google.dev/pricing
    # "Gemini": {
    #     "gemini-1.5-flash": 0.30,
    #     "gemini-1.5-pro": 10.50,
    # },
    #
    "Anthropic": {
        "claude-3-5-sonnet-20240620": 15.00,
        # "claude-3-opus-20240229": 75,
        # "claude-3-haiku-20240307": 1.25,
    },
    # model IDs from https://console.groq.com/docs/tool-use?hss_channel=tw-842860575289819136
    # prices not available yet, default to bedrock pricing
    # free tier: 30 rpm
    "Groq": {
        # "gemma2-9b-it": 0.22,
        # "llama-3.1-8b-instant": 0.22,
        "llama-3.1-70b-versatile": .99,
        # "llama-3.1-405b-reasoning": 16,
        # "mixtral-8x7b-32768": 0.7,
    },
    # from https://aws.amazon.com/bedrock/pricing/
    # model IDs from https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    "Bedrock": {
        # "mistral.mixtral-8x7b-instruct-v0:1": 0.7,
        # "mistral.mistral-large-2402-v1:0": 24,
        # "meta.llama3-1-8b-instruct-v1:0": 0.22,
        # "meta.llama3-1-70b-instruct-v1:0": 0.99,
        # "meta.llama3-1-405b-instruct-v1:0": 16,
    }
}

bandwidth_to_model = {
    "OpenAI": {
        "gpt-4o": 2,
        # "gpt-4o-mini": 1,
        # "gpt-3.5-turbo": 1,
        # "o1-preview": 1,
        # "o1-mini": 1,
    },
    # from https://ai.google.dev/pricing
    # "Gemini": {
    #     "gemini-1.5-flash": 1,
    #     "gemini-1.5-pro": 1,
    # },
    #
    "Anthropic": {
        "claude-3-5-sonnet-20240620": 2,
        # "claude-3-opus-20240229": 1,
        # "claude-3-haiku-20240307": 1,
    },
    # model IDs from https://console.groq.com/docs/tool-use?hss_channel=tw-842860575289819136
    # prices not available yet, default to bedrock pricing
    # free tier: 30 rpm
    "Groq": {
        # "gemma2-9b-it": 1,
        # "llama-3.1-8b-instant": 1,
        "llama-3.1-70b-versatile": 1,
        # "llama-3.1-405b-reasoning": 16,
        # "mixtral-8x7b-32768": 1,
    },
    # from https://aws.amazon.com/bedrock/pricing/
    # model IDs from https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    # "Bedrock": {
    #     "mistral.mixtral-8x7b-instruct-v0:1": 1,
    #     "mistral.mistral-large-2402-v1:0": 1,
    #     "meta.llama3-1-8b-instruct-v1:0": 1,
    #     "meta.llama3-1-70b-instruct-v1:0": 1,
        # "meta.llama3-1-405b-instruct-v1:0": 16,
    # }
}
