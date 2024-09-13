from cortext import ImageResponse, TextPrompting, StreamPrompting
from miner.providers import OpenAI, Anthropic, AnthropicBedrock, Groq, Gemini, Bedrock

task_image = ImageResponse.__name__
task_stream = StreamPrompting.__name__

openai_provider = OpenAI.__name__
anthropic_provider = Anthropic.__name__
anthropic_bedrock_provider = AnthropicBedrock.__name__
groq_provider = Groq.__name__
gemini_provider = Gemini.__name__
bedrock_provider = Bedrock.__name__

capacity_to_task_and_provider = {
    f"{task_image}_{openai_provider}": 1,
    f"{task_image}_{anthropic_provider}": 1,
    f"{task_image}_{anthropic_bedrock_provider}": 1,
    f"{task_image}_{groq_provider}": 1,
    f"{task_image}_{gemini_provider}": 1,
    f"{task_image}_{bedrock_provider}": 1,

    f"{task_stream}_{openai_provider}": 1,
    f"{task_stream}_{anthropic_provider}": 1,
    f"{task_stream}_{anthropic_bedrock_provider}": 1,
    f"{task_stream}_{groq_provider}": 1,
    f"{task_stream}_{gemini_provider}": 1,
    f"{task_stream}_{bedrock_provider}": 1,
}
