from abc import abstractmethod
import bittensor as bt
from typing import Tuple
import cortext
import time
import traceback
from collections import deque
from miner.config import config


class BaseService:
    def __init__(self, metagraph, request_timestamps, blacklist_amt):
        self.metagraph = metagraph
        self.request_timestamps = request_timestamps
        self.blacklist_amt = blacklist_amt if blacklist_amt is not None else config.BLACKLIST_AMT

    def base_blacklist(self, synapse) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            uid = None
            for _uid, _axon in enumerate(self.metagraph.axons):  # noqa: B007
                if _axon.hotkey == hotkey:
                    uid = _uid
                    break

            if uid is None and cortext.ALLOW_NON_REGISTERED is False:
                return True, f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}"

            # check the stake
            stake = self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]
            if stake < self.blacklist_amt:
                return True, f"Blacklisted a low stake {synapse_type} request: {stake} < {self.blacklist_amt} from {hotkey}"

            time_window = cortext.MIN_REQUEST_PERIOD * 60
            current_time = time.time()

            if hotkey not in self.request_timestamps:
                self.request_timestamps[hotkey] = deque()

            # Remove timestamps outside the current time window
            while self.request_timestamps[hotkey] and current_time - self.request_timestamps[hotkey][0] > time_window:
                self.request_timestamps[hotkey].popleft()

            # Check if the number of requests exceeds the limit
            if len(self.request_timestamps[hotkey]) >= cortext.MAX_REQUESTS:
                return (
                    True,
                    f"Request frequency for {hotkey} exceeded: "
                    f"{len(self.request_timestamps[hotkey])} requests in {cortext.MIN_REQUEST_PERIOD} minutes. "
                    f"Limit is {cortext.MAX_REQUESTS} requests."
                )

            self.request_timestamps[hotkey].append(current_time)

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")
