import os
import time
import argparse
import traceback
import bittensor as bt
from typing import Tuple, List, Generator, AsyncGenerator
import template
import openai
from fastapi.responses import StreamingResponse
from fastapi import FastAPI

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", default="my_custom_value")
    parser.add_argument("--netuid", type=int, default=1)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/miner")
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

class Miner:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.wallet, self.subtensor, self.metagraph, self.my_subnet_uid = self.initialize_wallet()
        self.axon = self.initialize_axon()

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")

    def initialize_wallet(self):
        wallet = bt.wallet(config=self.config)
        subtensor = bt.subtensor(config=self.config)
        metagraph = subtensor.metagraph(self.config.netuid)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(f"Your miner: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again.")
            exit()
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        return wallet, subtensor, metagraph, my_subnet_uid

    def initialize_axon(self):
        axon = bt.axon(wallet=self.wallet, config=self.config)
        axon.attach(
            forward_fn=self.process_question_endpoint,
            blacklist_fn=self.blacklist_fn,
            priority_fn=self.priority_fn
        )
        axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        axon.start()
        return axon

    def blacklist_fn(self, synapse: template.protocol.StreamingSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
        return False, "Hotkey recognized!"

    def priority_fn(self, synapse: template.protocol.StreamingSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def token_streamer(self, send: template.protocol.Send, chunks_gen: AsyncGenerator[str, None], synapse: template.protocol.StreamingSynapse):
        try:
            async for chunk in chunks_gen:
                # bt.logging.info(f"Sending chunk to client: {chunk}")
                await synapse.stream_output.put(chunk)  # Add chunk to the queue
                
                # Logging the chunk as it's added to the stream_output
                bt.logging.info(f"Added chunk to stream_output: {chunk}")
                
                await send({"type": "http.response.body", "body": chunk.encode('utf-8'), "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            await synapse.stream_output.put(None)  # Signal end of streaming
        except Exception as e:
            bt.logging.error(f"Error while streaming chunks: {e}")
            raise


    async def send_openai_request(self, prompt, engine) -> Generator[str, None, None]:
        try:
            bt.logging.info(f"sending openai request of {prompt} to {engine}")
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                stream=True
            )
            for chunk in response:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    chunk_message = str(delta['content'])
                    # bt.logging.info(f"Yielding chunk from OpenAI: {chunk_message}")
                    yield chunk_message
        except Exception as e:
            bt.logging.error(f"Got exception when calling openai {e}")
            traceback.print_exc()
            yield "Error calling model"

        def process_question_endpoint(self, synapse: StreamPrompting) -> StreamPrompting:
            try:
            bt.logging.info(f"Received synapse: {synapse}")
            prompt = synapse.streaming_input
            engine = synapse.streaming_engine
            bt.logging.info(f"Processing query from validator: '{prompt}' using {engine}")

            # Create the streaming response using the token_streamer function.
            chunks_gen = self.send_openai_request(prompt, engine)
            response = synapse.create_streaming_response(lambda send: self.token_streamer(send, chunks_gen, synapse))
            return response
        except Exception as e:
            bt.logging.error(f"Error in process_question_endpoint: {e}")
            raise e

    def run(self):
        step = 0
        while True:
            try:
                if step % 5 == 0:
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                step += 1
                time.sleep(1)
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception:
                bt.logging.error(traceback.format_exc())

if __name__ == "__main__":
    miner = Miner(get_config())
    miner.run()
