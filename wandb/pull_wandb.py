import wandb
import pandas as pd

api = wandb.Api()
username = "username here"
project_name = "project name here"
run_name = "run name here"
# run = api.run(f"/{username}/{project_name}/runs/{run_name}")
run = api.run("/cortex-t/synthetic-QA/runs/p0gdm9g7")
history = run.history()

# Write the history to a file in JSON format using pandas
history.to_json('run_history.json', orient='records', lines=True, indent=4)

print("History has been saved to run_history.json")

# api = wandb.Api()
# run = api.run("/cortex-t/synthetic-QA/runs/ycfmdj8t")
# print(run.history())

# import wandb
# import os

# # Replace with your run path "entity/project/run_id"
# run_path = "entity/project/run_id"

# # Initialize wandb API
# api = wandb.Api()

# # Access the run
# run = api.run("/cortex-t/synthetic-QA/runs/amczp753")

# # Create a directory to store images
# os.makedirs("wandb_images", exist_ok=True)

# # Iterate through all the files in the run
# for file in run.files():
#     # Check if the file is an image
#     if file.name.endswith(".png") or file.name.endswith(".jpg"):
#         # Download the file
#         file.download(root="wandb_images")
#         print(f"Downloaded {file.name}")

# print("All images have been downloaded to the 'wandb_images' directory.")
