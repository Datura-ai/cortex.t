from enum import Enum


class QueryType(str, Enum):  # Inherit from str to enforce the value type as string
    organic_type = 'organic'
    synthetic_type = 'synthetic'
