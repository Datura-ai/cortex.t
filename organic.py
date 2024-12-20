import sys
import aiohttp
import asyncio
import traceback
import random
import bittensor as bt
from aiohttp import ServerTimeoutError, ClientConnectorError
from bittensor import dendrite
from typing import List, Optional, Union, AsyncGenerator, Any, Dict, AsyncIterator
import pydantic
import time
from starlette.responses import StreamingResponse

class StreamPrompting(bt.StreamingSynapse):
    messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the StreamPrompting scenario, "
                    "each containing a role and content. Immutable.",
        allow_mutation=False,
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    seed: int = pydantic.Field(
        default="1234",
        title="Seed",
        description="Seed for text generation. This attribute is immutable and cannot be updated.",
    )

    temperature: float = pydantic.Field(
        default=0.0001,
        title="Temperature",
        description="Temperature for text generation. "
                    "This attribute is immutable and cannot be updated.",
    )

    max_tokens: int = pydantic.Field(
        default=2048,
        title="Max Tokens",
        description="Max tokens for text generation. "
                    "This attribute is immutable and cannot be updated.",
    )

    top_p: float = pydantic.Field(
        default=0.001,
        title="Top_p",
        description="Top_p for text generation. The sampler will pick one of "
                    "the top p percent tokens in the logit distirbution. "
                    "This attribute is immutable and cannot be updated.",
    )

    top_k: int = pydantic.Field(
        default=1,
        title="Top_k",
        description="Top_k for text generation. Sampler will pick one of  "
                    "the k most probablistic tokens in the logit distribtion. "
                    "This attribute is immutable and cannot be updated.",
    )

    completion: str = pydantic.Field(
        None,
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
                    "This attribute is mutable and can be updated.",
    )

    provider: str = pydantic.Field(
        default="OpenAI",
        title="Provider",
        description="The provider to use when calling for your response. "
                    "Options: OpenAI, Anthropic, Groq, Bedrock"
    )

    model: str = pydantic.Field(
        default="gpt-3.5-turbo",
        title="model",
        description="""
        The model to use when calling provider for your response.
        For Provider OpenAI:
         text_models = [
            "davinci-002",
            "gpt-4-1106-preview",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "babbage-002",
            "gpt-4",
            "gpt-4-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-4o",
            "gpt-4o-2024-05-13"
        ]
        For Provider Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
        For Provider Groq: gemma-7b-it, llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768
        For Provider Bedrock: anthropic.claude-3-sonnet-20240229-v1:0, cohere.command-r-v1:0, meta.llama2-70b-chat-v1,
         amazon.titan-text-express-v1, mistral.mistral-7b-instruct-v0:2
        last_updated = 17 June 2024
        """
    )

    uid: int = pydantic.Field(
        default=3,
        title="uid",
        description="The UID to send the streaming synapse to",
    )

    timeout: int = pydantic.Field(
        default=60,
        title="timeout",
        description="The timeout for the dendrite of the streaming synapse",
    )

    streaming: bool = pydantic.Field(
        default=True,
        title="streaming",
        description="whether to stream the output",
    )
    deserialize_flag: bool = pydantic.Field(
        default=True
    )
    task_id: str = pydantic.Field(
        default="9999",
        title="task_id",
        description="task id of the request from this syanpse."
    )
    validator_info: dict = pydantic.Field(
        default={},
        title="validator_info",
    )
    miner_info: dict = pydantic.Field(
        default={},
        title="miner_info",
    )
    time_taken: int = pydantic.Field(
        default=0,
        title="time_taken",
    )
    block_num: int = pydantic.Field(
        default=0,
        title="block_num",
    )
    cycle_num: int = pydantic.Field(
        default=0,
        title="cycle_num",
    )
    epoch_num: int = pydantic.Field(
        default=0,
        title="epoch num",
    )
    score: float = pydantic.Field(
        default=0,
        title="score",
    )
    similarity: float = pydantic.Field(
        default=0,
        title="similarity",
    )

    def to_headers(self) -> dict:
        headers = {"name": self.name, "timeout": str(self.timeout)}

        # Adding headers for 'axon' and 'dendrite' if they are not None
        if self.axon:
            headers.update(
                {
                    f"bt_header_axon_{k}": str(v)
                    for k, v in self.axon.dict().items()
                    if v is not None
                }
            )
        if self.dendrite:
            headers.update(
                {
                    f"bt_header_dendrite_{k}": str(v)
                    for k, v in self.dendrite.dict().items()
                    if v is not None
                }
            )

        headers[f"bt_header_input_obj_messages"] = "W10="
        headers["header_size"] = str(sys.getsizeof(headers))
        headers["total_size"] = str(self.get_total_size())
        headers["computed_body_hash"] = self.body_hash

        return headers

    async def process_streaming_response(self, response: StreamingResponse, organic=True) -> AsyncIterator[str]:
        if self.completion is None:
            self.completion = ""
        chunk_size = 100 if organic else 1024
        async for chunk in response.content.iter_chunked(chunk_size):
            tokens = chunk.decode("utf-8")
            self.completion += tokens
            yield tokens

    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix: str) -> dict[str, str]:
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "timeout": self.timeout,
            "streaming": self.streaming,
            "uid": self.uid,
        }


class CortexDendrite(dendrite):
    task_id = 0
    miner_to_session = {}

    def __init__(
            self, wallet: Optional[Union[bt.wallet, bt.Keypair]] = None
    ):
        super().__init__(wallet)

    async def call_stream(
            self,
            target_axon: Union[bt.AxonInfo, bt.axon],
            synapse: bt.StreamingSynapse = bt.Synapse(),  # type: ignore
            timeout: float = 12.0,
            deserialize: bool = True,
            organic: bool = True
    ) -> AsyncGenerator[Any, Any]:
        start_time = time.time()
        target_axon = (
            target_axon.info()
            if isinstance(target_axon, bt.axon)
            else target_axon
        )

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        endpoint = (
            f"0.0.0.0:{str(target_axon.port)}"
            if target_axon.ip == str(self.external_ip)
            else f"{target_axon.ip}:{str(target_axon.port)}"
        )
        url = f"http://{endpoint}/{request_name}"

        # Preprocess synapse for making a request
        synapse: StreamPrompting = self.preprocess_synapse_for_request(target_axon, synapse, timeout)  # type: ignore
        max_try = 0
        timeout = aiohttp.ClientTimeout(total=100, connect=timeout, sock_connect=timeout, sock_read=timeout)
        connector = aiohttp.TCPConnector(limit=200)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        try:
            while max_try < 3:
                async with session.post(
                        url,
                        headers=synapse.to_headers(),
                        json=synapse.dict(),
                ) as response:
                    # Use synapse subclass' process_streaming_response method to yield the response chunks
                    try:
                        async for chunk in synapse.process_streaming_response(response, organic):  # type: ignore
                            yield chunk  # Yield each chunk as it's processed
                    except aiohttp.client_exceptions.ClientPayloadError:
                        pass
                    except TimeoutError as err:
                        bt.logging.error(f"timeout error happens. max_try is {max_try}")
                        max_try += 1
                        continue
                    except ConnectionRefusedError as err:
                        bt.logging.error(f"can not connect to miner for now. connection failed")
                        break
                    except ClientConnectorError as err:
                        bt.logging.error(f"can not connect to miner for now. connection failed")
                        break
                    except ServerTimeoutError as err:
                        bt.logging.error(f"timeout error happens. max_try is {max_try}")
                        max_try += 1
                        continue
                    except Exception as err:
                        bt.logging.error(f"{err} issue from miner {synapse.uid} {synapse.provider} {synapse.model}")
                    finally:
                        pass
                break

        except Exception as e:
            bt.logging.error(f"{e} {traceback.format_exc()}")
        finally:
            synapse.dendrite.process_time = str(time.time() - start_time)
            await session.close()

    async def call_stream_in_batch(
            self,
            target_axons: List[Union[bt.AxonInfo, bt.axon]],
            synapses: List[bt.StreamingSynapse] = bt.Synapse(),  # type: ignore
            timeout: float = 12.0,
            deserialize: bool = True,
    ) -> AsyncGenerator[Any, Any]:
        pass


async def generate_prompts(num_prompts=100):
    subjects = [
        "artificial intelligence",
        "climate change",
        "space exploration",
        "quantum computing",
        "renewable energy",
        "virtual reality",
        "biotechnology",
        "cybersecurity",
        "autonomous vehicles",
        "blockchain",
        "3D printing",
        "robotics",
        "nanotechnology",
        "gene editing",
        "Internet of Things",
        "augmented reality",
        "machine learning",
        "sustainable agriculture",
        "smart cities",
        "digital privacy",
        "wearable technology",
        "artificial neural networks",
        "drone technology",
        "digital currencies",
        "supercomputers",
        "genetic engineering",
        "precision medicine",
        "brain-computer interfaces",
        "quantum cryptography",
        "carbon capture technologies",
        "smart manufacturing",
        "biometrics",
        "edge computing",
        "cloud computing",
        "personalized healthcare",
        "cryptocurrency mining",
        "5G networks",
        "autonomous drones",
        "robot-assisted surgery",
        "big data analytics",
        "energy storage",
        "quantum supremacy",
        "self-driving trucks",
        "AI ethics",
        "distributed computing",
        "exoskeleton technology",
        "carbon-neutral technologies",
        "food security",
        "telemedicine",
        "smart grids",
        "renewable water resources",
    ]

    prompt_types = [
        "Explain the concept of {}.",
        "Discuss the potential impact of {}.",
        "Compare and contrast two approaches to {}.",
        "Outline the future prospects of {}.",
        "Describe the ethical implications of {}.",
        "Analyze the current state of {}.",
        "Propose a solution using {}.",
        "Evaluate the pros and cons of {}.",
        "Predict how {} will change in the next decade.",
        "Discuss the role of {} in solving global challenges.",
        "Explain how {} is transforming industry.",
        "Describe a day in the life with advanced {}.",
        "Outline the key challenges in developing {}.",
        "Discuss the intersection of {} and another field.",
        "Explain the historical development of {}.",
        "What are the key innovations in {}?",
        "How can {} be used to improve everyday life?",
        "What are the societal implications of {}?",
        "What are the business applications of {}?",
        "Discuss how {} could evolve in the next 50 years.",
        "What are the limitations of current {} technologies?",
        "How does {} compare to previous technologies?",
        "What are the most exciting breakthroughs in {}?",
        "What role does {} play in education?",
        "What are the security concerns related to {}?",
        "How can {} be integrated into future urban planning?",
        "What role will {} play in the future of transportation?",
        "How does {} affect personal privacy?",
        "Discuss the environmental impacts of {}.",
        "How is {} reshaping healthcare?",
        "What role does {} play in data security?",
        "What are the political implications of {}?",
        "What are the major regulatory hurdles for {}?",
        "What role will {} play in space exploration?",
        "How can {} address global inequality?",
        "Discuss the relationship between {} and creativity.",
        "How can {} be used in disaster management?",
        "What future careers might emerge due to {}?",
        "What is the relationship between {} and economics?",
        "How will {} impact future generations?",
        "How can governments promote the use of {}?",
    ]

    # Calculate the new maximum possible prompts
    max_possible_prompts = len(subjects) * len(prompt_types)
    num_prompts = min(num_prompts, max_possible_prompts)

    prompts = set()

    while len(prompts) < num_prompts:
        subject = random.choice(subjects)
        prompt_type = random.choice(prompt_types)
        prompt = prompt_type.format(subject)
        prompts.add(prompt)

    return list(prompts)


async def query_miner(dendrite: CortexDendrite, axon_to_use, synapse, timeout=60, streaming=True):
    try:
        print(f"calling vali axon {axon_to_use} to miner uid {synapse.uid} for query {synapse.messages}")
        resp = dendrite.call_stream(
            target_axon=axon_to_use,
            synapse=synapse,
            timeout=timeout
        )
        return await handle_response(resp)
    except Exception as e:
        print(f"Exception during query: {traceback.format_exc()}")
        return None


async def handle_response(resp):
    full_response = ""
    try:
        async for chunk in resp:
            if isinstance(chunk, str):
                full_response += chunk
                print(chunk, end='', flush=True)
            else:
                print(f"\n\nFinal synapse: {chunk}\n")
    except Exception as e:
        print(f"Error processing response for uid {e}")
    return full_response


async def main():
    print("synching metagraph, this takes way too long.........")
    subtensor = bt.subtensor(network="finney")
    meta = subtensor.metagraph(netuid=18)
    print("metagraph synched!")

    # This needs to be your validator wallet that is running your subnet 18 validator
    # wallet = bt.wallet(name="default", hotkey="default")
    wallet = bt.wallet(name="default", hotkey="default")
    dendrite = CortexDendrite(wallet=wallet)
    vali_uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    axon_to_use = meta.axons[vali_uid]
    print(f"axon to use: {axon_to_use}")

    num_prompts = 50
    prompts = await generate_prompts(num_prompts)
    synapses = [StreamPrompting(
        messages=[{"role": "user", "content": prompt}],
        provider="OpenAI",
        model="gpt-4o"
    ) for prompt in prompts]
    # synapses = [StreamPrompting(
    #     messages=[{"role": "user", "content": prompt}],
    #     provider="Anthropic",
    #     model="claude-3-5-sonnet-20240620"
    # ) for prompt in prompts]
    # synapses = [StreamPrompting(
    #     messages=[{"role": "user", "content": prompt}],
    #     provider="Groq",
    #     model="llama-3.1-70b-versatile"
    # ) for prompt in prompts]

    async def query_and_log(synapse):
        return await query_miner(dendrite, axon_to_use, synapse)

    responses = await asyncio.gather(*[query_and_log(synapse) for synapse in synapses])

    import csv
    with open('miner_responses.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Prompt', 'Response'])
        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response])

    print("Responses saved to miner_responses.csv")


if __name__ == "__main__":
    asyncio.run(main())

