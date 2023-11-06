<<<<<<< HEAD
alias start_miner1='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name 1 --wallet.hotkey 1 --axon.port 15000 --logging.trace'
alias start_miner2='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name 1 --wallet.hotkey 2 --axon.port 9001 --logging.trace'
alias start_miner3='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name 1 --wallet.hotkey 3 --axon.port 9002 --logging.trace'
alias start_vali='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/validator.py --netuid 18 --subtensor.network finney --wallet.name default --wallet.hotkey default --logging.trace'

# cd bittensor-subnet-template && pip install -e . && pip3 install openai wandb && wandb init && echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc
=======
alias start_miner1='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name im --wallet.hotkey 01 --axon.port 9000 --logging.debug'
alias start_miner2='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name im --wallet.hotkey 02 --axon.port 9001 --logging.debug'
alias start_miner3='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name 1 --wallet.hotkey 3 --axon.port 9002 --logging.debug'
alias start_miner4='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/miner.py --netuid 18 --subtensor.network finney --wallet.name 1 --wallet.hotkey 4 --axon.port 9003 --logging.trace'
alias start_vali='python3 /root/subnets/exploitproof-net/bittensor-subnet-template/neurons/validator.py --netuid 18 --subtensor.network finney --wallet.name default --wallet.hotkey default --logging.debug'
>>>>>>> new-branch-name
