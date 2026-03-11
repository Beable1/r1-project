import sys
sys.path.append('/home/beable/lerobot/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('Beable/dexterious_ee_test')
print(f"Dataset length: {len(ds)}")

for i in [0, 944, 13501, 27000]:
    try:
        sample = ds[i]
        print(f"Sample {i} successfully loaded. image shape: {sample['observation.images.rgb'].shape}, Index in ep: {sample['frame_index']}")
    except Exception as e:
        print(f"Sample {i} failed: {e}")
