#!/usr/bin/env python3
"""
DreamerV3 Quick Start Example
============================

This script demonstrates how to quickly get started with DreamerV3 training.
It uses the simplest possible setup with default configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import gymnasium as gym
from models.dreamer_v3_agent import DreamerV3Agent
from envs.minigrid_env import MiniGridWrapper
from utils.dreamer_buffer import DreamerBuffer

def main():
    print("🚀 DreamerV3 Quick Start")
    print("=" * 50)
    
    # Environment setup
    env_name = "MiniGrid-Empty-8x8-v0"
    env = MiniGridWrapper(env_name)
    
    print(f"Environment: {env_name}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Agent setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    agent = DreamerV3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )
    
    # Buffer setup
    buffer = DreamerBuffer(capacity=50000, obs_shape=env.observation_space.shape)
    
    print("\n✅ Setup complete!")
    print("\nTo start training:")
    print("1. Run: python scripts/train_dreamer_v3.py")
    print("2. Or: python scripts/train_dreamer_v3_with_checkpointing.py")
    print("3. Monitor with WandB dashboard")
    
    print("\n🧪 To test all upgrades:")
    print("python scripts/test_dreamer_v3_upgrades.py")
    
    print("\n🏠 For complex environments:")
    print("python scripts/train_dreamer_v3_fourrooms_weekend.py")
    
    print("\n🎯 Expected results:")
    print("- MiniGrid-Empty: 90-100% success rate")
    print("- MiniGrid-FourRooms: 5%+ success rate")
    
if __name__ == "__main__":
    main() 