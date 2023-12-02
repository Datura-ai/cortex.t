import subprocess

num_miners = 10
base_port = 5002
pm2_commands = []

for i in range(num_miners):
    port = base_port + i

    # Construct the PM2 start command
    command = f"pm2 start miner.py --interpreter python3 --name miner{i} -- --wallet.name 1 --wallet.hotkey {i+1} --subtensor.network test --netuid 24 --axon.port {port}"
    pm2_commands.append(command)

# Execute all commands
for cmd in pm2_commands:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
