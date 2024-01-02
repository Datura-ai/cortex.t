import abc
import subprocess
import time

import pytest


class ActiveSubnetworkBaseTest(abc.ABC):

    miner_process = None
    validator_process = None

    @classmethod
    @abc.abstractmethod
    def validator_path_and_args(cls) -> list[str]:
        """
        Return the path to validator start script and all arguments required to start it, e.g.
        `validator.py --opt1=8 --opt2=9`
        """
        ...

    @classmethod
    @abc.abstractmethod
    def miner_path_and_args(cls) -> list[str]:
        """
        Return the path to miner start script and all arguments required to start it, e.g.
        `miner.py --opt1=8 --opt2=9`
        """
        ...

    @classmethod
    @abc.abstractmethod
    def check_if_validator_is_up(cls) -> bool:
        """
        Check if the validator process is up and ready to accept requests. This can be achieved by making a network,
        reading logs etc.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def check_if_miner_is_up(cls) -> bool:
        """
        Check if the miner process is up and ready to accept requests. This can be achieved by making a network, reading
        logs etc.
        """
        ...

    @classmethod
    def start_process(cls, args):
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def wait_for_process_start(cls, process_name, probe_function, process: subprocess.Popen):
        for i in range(300):
            if probe_function():
                return
            if process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f'Process {process_name} did not start\n'
                           f'stdout={process.stdout.read()}\n\n'
                           f'stderr={process.stderr.read()}')

    @classmethod
    @pytest.fixture(autouse=True, scope="session")
    def start_validator_and_miner(cls):
        cls.miner_process = cls.start_process(cls.miner_path_and_args())
        cls.validator_process = cls.start_process(cls.validator_path_and_args())
        cls.wait_for_process_start('validator', cls.check_if_validator_is_up, cls.validator_process)
        cls.wait_for_process_start('miner', cls.check_if_miner_is_up, cls.miner_process)
        yield
        if cls.miner_process:
            cls.miner_process.send_signal(9)
        if cls.validator_process:
            cls.validator_process.send_signal(9)
