from typing import Union
from .prompt import PromptService
from .image import ImageService
from .embedding import EmbeddingService
from .check_status import IsAliveService
from .capacity import CapacityService

ALL_SERVICE_TYPE = Union[PromptService, ImageService, EmbeddingService, IsAliveService, CapacityService]
__all__ = [PromptService, ImageService, EmbeddingService, CapacityService, ALL_SERVICE_TYPE]
