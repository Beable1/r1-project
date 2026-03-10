
import torch
import os
import sys

# Add Isaac Lab paths if needed, or assume environment is set
sys.path.append("/home/beable/IsaacLab/IsaacLab")

try:
    model_path = "/home/beable/Desktop/r1-project/locomotion-models/2026-02-06_20-29-29/model_73850.pt"
    print(f"Loading model from {model_path}")
    
    # Load with map_location
    loaded_dict = torch.load(model_path, map_location='cpu')
    
    print("Keys in loaded dict:", loaded_dict.keys())
    
    if 'model_state_dict' in loaded_dict:
        state_dict = loaded_dict['model_state_dict']
        print("\nModel State Dict Keys (first 10):")
        for k in list(state_dict.keys())[:10]:
            print(k)
            
        # Try to infer input/output shape from weights
        # Usually 'actor.0.weight' or 'std' or similar
    
    if 'optimizer_state_dict' in loaded_dict:
        print("\nHas optimizer state")
        
except Exception as e:
    print(f"Error loading model: {e}")
