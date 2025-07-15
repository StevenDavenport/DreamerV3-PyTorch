import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

class MiniGridEnvWrapper:
    """Wrapper for MiniGrid environments for RSSM testing."""
    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0", render: bool = False):
        self.env = gym.make(env_name, render_mode="human" if render else None)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = render

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        return self._preprocess_observation(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._preprocess_observation(obs), reward, done, info

    def _preprocess_observation(self, obs: Dict) -> np.ndarray:
        # MiniGrid observations are dicts with 'image' key (3D array)
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image'].astype(np.float32) / 10.0  # Normalize
            return img.flatten()
        elif isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
        else:
            return np.array(obs, dtype=np.float32).flatten()

    def sample_action(self) -> int:
        return self.action_space.sample()

    def get_action_dim(self) -> int:
        return self.action_space.n

    def get_obs_dim(self) -> int:
        # Use flattened image size
        obs = self.reset()
        return obs.size

    def close(self):
        self.env.close() 