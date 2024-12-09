from pydantic import BaseModel, Field
from datetime import datetime
import enum
from typing import Optional, Any

from pydantic import BaseModel
from pydantic.fields import Field


class Role(str, enum.Enum):
    """Message is sent by which role?"""

    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role = Role.user
    content: str = Field(default=..., examples=["Remind me that I have forgot to set the messages"])

    class Config:
        use_enum_values = True


class ChatRequest(BaseModel):
    messages: list[Message] = Field(...)
    temperature: float = Field(default=0.0001, examples=[0.5, 0.4, 0.3], title="Temperature",
                               description="Temperature for text generation.")
    max_tokens: int = Field(2048, title="Max Tokens", description="Max tokens for text generation.")
    model: str = Field(default="gpt-4o", examples=["gpt-4o"], title="Model")
    provider: str = Field(default="OpenAI", examples=["OpenAI"], title="Provider")
    top_p: float = Field(default=0.001, title="Top P", description="Top P for text generation.")
    stream: bool = Field(default=True, title="Stream", description="Stream for text generation.")
    logprobs: bool = True

