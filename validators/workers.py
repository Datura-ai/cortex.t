import bittensor as bt
from cortext import REDIS_RESULT_STREAM, REDIS_RESULT
from validators.utils import get_redis_client, error_handler


class Worker:

    def __init__(self, synapse, dendrite, axon):
        self.synapse = synapse
        self.dendrite = dendrite
        self.axon = axon

    @error_handler
    async def run_task(self):
        # Pull task from worker-specific queue
        redis_client = get_redis_client()
        task_id = self.synapse.task_id
        bt.logging.trace(f"Worker {task_id} received task: {self.synapse}")

        await self.dendrite.aclose_session()
        responses = await self.dendrite(
            axons=[self.axon],
            synapse=self.synapse,
            deserialize=self.synapse.deserialize_flag,
            timeout=self.synapse.timeout,
            streaming=self.synapse.streaming,
        )
        if self.synapse.streaming:
            async for chunk in responses[0]:
                if isinstance(chunk, str):
                    redis_client.xadd(REDIS_RESULT_STREAM + f"{task_id}", {"chunk": chunk})
        else:
            redis_client.rpush(REDIS_RESULT, responses[0])
        redis_client.close()
