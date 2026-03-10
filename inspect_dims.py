
import torch
import sys

try:
    model_path = "/home/beable/Desktop/r1-project/locomotion-models/2026-02-06_20-29-29/model_73850.pt"
    loaded_dict = torch.load(model_path, map_location='cpu')
    state_dict = loaded_dict['model_state_dict']
    
    print("Layer Shapes:")
    for key in ['actor.0.weight', 'actor.0.bias', 'actor.6.weight', 'actor.6.bias']:
        if key in state_dict:
            print(f"{key}: {state_dict[key].shape}")
            
except Exception as e:
    print(f"Error: {e}")
