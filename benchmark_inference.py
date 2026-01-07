#!/usr/bin/env python3
"""
Benchmark script for Diffusion Policy inference speed
"""

import torch
import numpy as np
import time
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE

# Model path
MODEL_PATH = "/home/beable/lerobot/outputs/train/diffusion_dexterous_fullres2/checkpoints/last/pretrained_model"

# Config
DEVICE = "cuda"
N_WARMUP = 5
N_BENCHMARK = 50

def main():
    print("=" * 60)
    print("🚀 DIFFUSION POLICY INFERENCE BENCHMARK")
    print("=" * 60)
    
    # Load model
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    policy = DiffusionPolicy.from_pretrained(MODEL_PATH)
    policy.to(DEVICE)
    policy.eval()
    
    n_obs_steps = policy.config.n_obs_steps
    n_action_steps = policy.config.n_action_steps
    
    print(f"✅ Model loaded!")
    print(f"   n_obs_steps: {n_obs_steps}")
    print(f"   n_action_steps: {n_action_steps}")
    print(f"   Device: {DEVICE}")
    
    # Create dummy input
    # State: (1, n_obs_steps, 17)
    # Images: (1, n_obs_steps, 1, 3, 360, 640)
    state = torch.randn(1, n_obs_steps, 17, device=DEVICE, dtype=torch.float32)
    images = torch.randn(1, n_obs_steps, 1, 3, 360, 640, device=DEVICE, dtype=torch.float32)
    
    batch = {
        OBS_STATE: state,
        OBS_IMAGES: images,
    }
    batch = policy.normalize_inputs(batch)
    
    print(f"\n📊 Input shapes:")
    print(f"   State: {state.shape}")
    print(f"   Images: {images.shape}")
    
    # Warmup
    print(f"\n🔥 Warming up ({N_WARMUP} runs)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = policy.diffusion.generate_actions(batch)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"\n⏱️  Benchmarking ({N_BENCHMARK} runs)...")
    times = []
    
    with torch.no_grad():
        for i in range(N_BENCHMARK):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            actions = policy.diffusion.generate_actions(batch)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
    
    # Stats
    times = np.array(times)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    
    # Effective Hz considering action chunking
    # One inference produces n_action_steps actions
    # If control loop runs at 10Hz, one inference covers n_action_steps * 100ms
    effective_hz = 1000.0 / mean_ms  # raw inference Hz
    
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)
    print(f"\n🕐 Inference Time (per diffusion call):")
    print(f"   Mean:  {mean_ms:.2f} ms")
    print(f"   Std:   {std_ms:.2f} ms")
    print(f"   Min:   {min_ms:.2f} ms")
    print(f"   Max:   {max_ms:.2f} ms")
    
    print(f"\n⚡ Inference Speed:")
    print(f"   Raw:   {effective_hz:.1f} Hz")
    print(f"   Actions per inference: {n_action_steps}")
    print(f"   Effective control rate: {effective_hz * n_action_steps:.1f} Hz")
    
    print(f"\n💡 At 10Hz control loop:")
    print(f"   New inference every: {n_action_steps / 10:.1f} seconds")
    print(f"   Real-time capable: {'✅ YES' if mean_ms < (n_action_steps * 100) else '❌ NO'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
