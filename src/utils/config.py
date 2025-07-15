"""
Configuration management for the skill-conditioned MBRL system.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from omegaconf import DictConfig, OmegaConf
import os


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic)")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=100, description="Maximum tokens for responses")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")


class WorldModelConfig(BaseModel):
    """Configuration for the RSSM world model."""
    latent_dim: int = Field(default=256, description="Latent state dimension")
    hidden_dim: int = Field(default=512, description="Hidden layer dimension")
    num_layers: int = Field(default=3, description="Number of layers")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    kl_weight: float = Field(default=1.0, description="KL divergence weight")


class PolicyConfig(BaseModel):
    """Configuration for the skill-conditioned policy."""
    hidden_dim: int = Field(default=512, description="Hidden layer dimension")
    num_layers: int = Field(default=3, description="Number of layers")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    skill_embedding_dim: int = Field(default=768, description="Skill embedding dimension")


class FeasibilityConfig(BaseModel):
    """Configuration for goal feasibility scoring."""
    hidden_dim: int = Field(default=512, description="Hidden layer dimension")
    num_layers: int = Field(default=2, description="Number of layers")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    num_candidates: int = Field(default=5, description="Number of goal candidates to score")


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    batch_size: int = Field(default=32, description="Training batch size")
    buffer_size: int = Field(default=100000, description="Replay buffer size")
    num_episodes: int = Field(default=1000, description="Number of training episodes")
    episode_length: int = Field(default=100, description="Maximum episode length")
    imagination_steps: int = Field(default=50, description="Steps to imagine ahead")
    update_frequency: int = Field(default=4, description="Policy update frequency")
    target_update_frequency: int = Field(default=100, description="Target network update frequency")


class EnvironmentConfig(BaseModel):
    """Configuration for the environment."""
    name: str = Field(default="crafter", description="Environment name")
    observation_dim: int = Field(default=64, description="Observation dimension")
    action_dim: int = Field(default=17, description="Action dimension")
    render: bool = Field(default=False, description="Whether to render episodes")


class LoggingConfig(BaseModel):
    """Configuration for logging and monitoring."""
    log_dir: str = Field(default="./logs", description="Logging directory")
    wandb_project: Optional[str] = Field(default=None, description="Weights & Biases project name")
    log_frequency: int = Field(default=100, description="Logging frequency")
    save_frequency: int = Field(default=1000, description="Model save frequency")


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    seed: int = Field(default=42, description="Random seed")
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")
    num_workers: int = Field(default=4, description="Number of data loading workers")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load with OmegaConf
    conf = OmegaConf.load(config_path)
    
    # Convert to Pydantic model
    config_dict = OmegaConf.to_container(conf, resolve=True)
    return Config(**config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = config.dict()
    conf = OmegaConf.create(config_dict)
    OmegaConf.save(conf, config_path)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# Example usage
if __name__ == "__main__":
    config = get_default_config()
    print(config.json(indent=2)) 