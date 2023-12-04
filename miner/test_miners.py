import subprocess

num_miners = 35
base_port = 5002
start_num = 5 
pm2_commands = []

for i in range(num_miners):
    miner_number = start_num + i
    port = base_port + i

    # Construct the PM2 start command
    command = f"pm2 start miner.py --interpreter python3 --name {miner_number} -- --wallet.name 1 --wallet.hotkey {miner_number} --subtensor.network test --netuid 24 --axon.port {port}"
    pm2_commands.append(command)

# Execute all commands
for cmd in pm2_commands:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
