from enum import Enum


class BandWidth(Enum):
    BATCH_SIZE = "batch_size"
    TASKS_PER_SEC = "tasks_per_sec"
    CHARS_PER_SEC = "chars_per_sec"
    IO_NETWORK_PER_SEC = "bytes_per_sec"
