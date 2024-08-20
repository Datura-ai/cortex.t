#!/bin/bash

read -p "Enter the subtensor.network argument [test]: " network
network=${network:-test }

read -p "Enter the netuid argument [24]: " netuid
netuid=${netuid:-24}

read -p "Enter the wallet.name argument [validator]: " wallet_name
wallet_name=${wallet_name:-validator}

read -p "Enter the wallet.hotkey argument [default]: " wallet_hotkey
wallet_hotkey=${wallet_hotkey:-default}

read -p "you want disable wandb [true]: " wandb_off
wandb_off=${wandb_off:-true}

read -p "you want enable tracing [true]: " tracing
tracing=${tracing:-true}

command_to_run="python3 -m validators.validator --subtensor.network $network --netuid $netuid --wallet.name $wallet_name --wallet.hotkey $wallet_hotkey"

if [ "$tracing" = "true" ]; then
    command_to_run="$command_to_run --logging.trace"
fi

if [ "$wandb_off" = "true" ]; then
    command_to_run="$command_to_run --wandb_off"
fi

# Run the Python script with the provided arguments
echo $command_to_run
$command_to_run "$@"
