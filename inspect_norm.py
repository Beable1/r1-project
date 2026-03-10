
import torch

try:
    model_path = "/home/beable/Desktop/r1-project/locomotion-models/2026-02-06_20-29-29/model_73850.pt"
    loaded_dict = torch.load(model_path, map_location='cpu')
    state_dict = loaded_dict['model_state_dict']
    
    print("Keys checking:")
    if 'mean' in state_dict:
        print(f"Found 'mean', shape: {state_dict['mean'].shape}")
    else:
        print("'mean' NOT found")
        
    if 'std' in state_dict:
        print(f"Found 'std', shape: {state_dict['std'].shape}")
    else:
        print("'std' NOT found")
        
    # Check for other potential normalization keys (running_mean, etc)
    for k in state_dict.keys():
        if 'mean' in k or 'norm' in k or 'std' in k:
            print(f"Match: {k}")

except Exception as e:
    print(f"Error: {e}")
