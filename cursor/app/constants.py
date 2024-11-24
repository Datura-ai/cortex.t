llm_models = [
    {
        "id": "chat-gpt-4o",
        "name": "Meta: gpt-4o Instruct",
        "created": 8192,
        "description": "OpenAI gpt-4o model.",
        "context_length": 8192,
        "architecture": {"modality": "text->text", "tokenizer": "gpt-4o", "instruct_type": "gpt-4o"},
        "pricing": {"prompt": "0.000000001", "completion": "0.000000001", "image": "0", "request": "0"},
    },
    {
        "id": "chat-claude-3-5-sonnet-20240620",
        "name": "Meta: claude-3-5-sonnet-20240620 Instruct",
        "created": 8192,
        "description": "claude-3-5-sonnet-20240620.",
        "context_length": 8192,
        "architecture": {"modality": "text->text", "tokenizer": "claude-3-5-sonnet-20240620", "instruct_type": "claude-3-5-sonnet-20240620"},
        "pricing": {"prompt": "0.000000001", "completion": "0.000000001", "image": "0", "request": "0"},
    },
    {
        "id": "chat-llama-3.1-70b-versatile",
        "name": "Meta: llama-3.1-70b-versatile Instruct",
        "created": 8192,
        "description": "llama-3.1-70b-versatile.",
        "context_length": 8192,
        "architecture": {"modality": "text->text", "tokenizer": "llama-3.1-70b-versatile", "instruct_type": "llama-3.1-70b-versatile"},
        "pricing": {"prompt": "0.00000001", "completion": "0.000000001", "image": "0", "request": "0"},
    }
]