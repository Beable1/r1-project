import os
import sys
sys.path.append('/home/beable/lerobot/src')

from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("Loading dataset from local directory...")
ds = LeRobotDataset(
    repo_id="Beable/dexterious_ee_test",
    root="/home/beable/Desktop/r1-project/output/"
)

print(f"Dataset loaded. Total episodes: {ds.num_episodes}, Total frames: {ds.num_frames}")

print("Pushing dataset to Hugging Face Hub (this may take a few minutes)...")
ds.push_to_hub()
print("Success! Dataset is now available on Hugging Face.")
