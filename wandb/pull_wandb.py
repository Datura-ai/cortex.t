import wandb
import pandas as pd

api = wandb.Api()
username = "username here"
project_name = "project name here"
run_name = "run name here"
run = api.run(f"/{username}/{project_name}/runs/{run_name}")
history = run.history()

# Write the history to a file in JSON format using pandas
history.to_json('run_history.json', orient='records', lines=True, indent=4)

print("History has been saved to run_history.json")
