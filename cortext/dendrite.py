from typing import Union, AsyncGenerator, Any

import aiohttp
import bittensor as bt
from aiohttp import ServerTimeoutError
from bittensor import dendrite
import traceback
import time
from typing import Optional, List

from cortext import StreamPrompting


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
        session = CortexDendrite.miner_to_session.get(url)
        try:
            while max_try < 3:
                if not session:
                    timeout = aiohttp.ClientTimeout(total=300, connect=timeout, sock_connect=timeout, sock_read=timeout)
                    connector = aiohttp.TCPConnector(limit=200)
                    session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                    CortexDendrite.miner_to_session[url] = session
                async with session.post(
                        url,
                        headers=synapse.to_headers(),
                        json=synapse.dict(),
                ) as response:
                    # Use synapse subclass' process_streaming_response method to yield the response chunks
                    try:
                        async for chunk in synapse.process_streaming_response(response):  # type: ignore
                            yield chunk  # Yield each chunk as it's processed
                    except aiohttp.client_exceptions.ClientPayloadError:
                        pass
                    except TimeoutError as err:
                        bt.logging.error(f"timeout error happens. max_try is {max_try}")
                        max_try += 1
                        continue
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

    async def call_stream_in_batch(
            self,
            target_axons: List[Union[bt.AxonInfo, bt.axon]],
            synapses: List[bt.StreamingSynapse] = bt.Synapse(),  # type: ignore
            timeout: float = 12.0,
            deserialize: bool = True,
    ) -> AsyncGenerator[Any, Any]:
        pass
