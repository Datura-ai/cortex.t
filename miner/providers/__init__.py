import groq

from miner.providers.open_ai import OpenAI
from cortext import ALL_SYNAPSE_TYPE


def get_provider(service_name: str, synapse: ALL_SYNAPSE_TYPE):
    if service_name == 'open_ai':
        return OpenAI(synapse=synapse)
    if service_name == 'qroq':
        return groq.Groq
