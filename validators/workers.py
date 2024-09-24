import bittensor as bt
from cortext import REDIS_RESULT_STREAM, REDIS_RESULT


class Worker:

    def __init__(self, synapse, dendrite, axon, redis_client):
        self.redis_client = redis_client
        self.synapse = synapse
        self.dendrite = dendrite
        self.axon = axon

    async def run_task(self):
        # Pull task from worker-specific queue
        task_id = self.synapse.task_id
        bt.logging.trace(f"Worker {task_id} received task: {self.synapse}")
        try:
            responses = await self.dendrite(
                axons=[self.axon],
                synapse=self.synapse,
                deserialize=self.synapse.deserialize_flag,
                timeout=self.synapse.timeout,
                streaming=self.synapse.streaming,
            )
        except Exception as err:
            bt.logging.exception(err)
        else:
            if self.synapse.streaming:
                async for chunk in responses[0]:
                    if isinstance(chunk, str):
                        await self.redis_client.xadd(REDIS_RESULT_STREAM + f"{task_id}", {"chunk": chunk})
            else:
                await self.redis_client.rpush(REDIS_RESULT, responses[0])
