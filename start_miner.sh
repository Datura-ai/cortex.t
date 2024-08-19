#!/bin/bash

read -p "Enter the subtensor.network argument [test]: " arg1
arg1=${arg1:-test }

read -p "Enter the netuid argument [24]: " arg2
arg2=${arg2:-24}

read -p "Enter the wallet.name argument [validator]: " arg3
arg3=${arg3:-validator}

read -p "Enter the wallet.hotkey argument [default]: " arg4
arg4=${arg4:-default}

# Run the Python script with the provided arguments
python3 -m miner.miner --subtensor.network $arg1 --netuid $arg2 --wallet.name $arg3 --wallet.hotkey $arg4
