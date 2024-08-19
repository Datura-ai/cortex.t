from typing import Union
from .prompt import PromptService
from .image import ImageService
from .embedding import EmbeddingService
from .text import TextService
from .check_status import IsAliveService

ALL_SERVICE_TYPE = Union[PromptService, ImageService, EmbeddingService, TextService, IsAliveService]
__all__ = [PromptService, ImageService, EmbeddingService, ALL_SERVICE_TYPE]
