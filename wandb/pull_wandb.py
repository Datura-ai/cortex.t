import wandb
import pandas as pd

api = wandb.Api()
run = api.run("/surcyf2/openai_qa/runs/wev5by5x")
history = run.history()

# Write the history to a file in JSON format using pandas
history.to_json('run_history.json', orient='records', lines=True, indent=4)

print("History has been saved to run_history.json")
