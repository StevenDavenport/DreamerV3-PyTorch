"""
DreamerV3-style RSSM Agent with robust reward prediction and training.
Incorporates all 4 key upgrades from DreamerV3 to handle sparse rewards and data imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

from models.world_model import RSSMWorldModel
from models.dreamer_v3_utils import (
    SymlogTwoHotRewardPredictor, 
    SymlogTwoHotCritic,
    ReturnNormalizer,
    free_bits_kl_loss,
    kl_balancing_loss,
    symlog,
    symexp
)


class DreamerV3Agent(nn.Module):
    """
    RSSM Agent with DreamerV3 improvements for robust imagination-based RL.
    
    Key improvements:
    1. Symlog + Two-hot encoding for reward/value prediction
    2. Percentile-based return normalization 
    3. Free bits mechanism for KL loss
    4. KL balancing for stable world model learning
    """
    
    def __init__(self, obs_dim: int, action_dim: int, rssm_hidden_dim: int, rssm_latent_dim: int,
                 policy_hidden_dim: int = 256, value_hidden_dim: int = 256, 
                 reward_hidden_dim: int = 256, continue_hidden_dim: int = 256,
                 num_reward_bins: int = 41, num_value_bins: int = 41):
        super().__init__()
        
        # Core dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = rssm_latent_dim
        self.hidden_dim = rssm_hidden_dim
        
        # World model (RSSM)
        self.rssm = RSSMWorldModel(obs_dim, action_dim, rssm_hidden_dim, rssm_latent_dim)
        
        # DreamerV3 Policy head (simple categorical for discrete actions)
        self.policy_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim)
        )
        
        # DreamerV3 Value head with symlog + two-hot encoding
        self.value_net = SymlogTwoHotCritic(
            input_dim=rssm_latent_dim,
            hidden_dim=value_hidden_dim,
            num_bins=num_value_bins
        )
        
        # DreamerV3 Reward model with symlog + two-hot encoding
        self.reward_net = SymlogTwoHotRewardPredictor(
            input_dim=rssm_latent_dim,
            hidden_dim=reward_hidden_dim,
            num_bins=num_reward_bins
        )
        
        # Continue model (unchanged for now)
        self.continue_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, continue_hidden_dim),
            nn.ReLU(),
            nn.Linear(continue_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # DreamerV3 Return normalizer
        self.return_normalizer = ReturnNormalizer(
            percentile_low=5.0,
            percentile_high=95.0,
            ema_alpha=0.99,
            min_threshold=1.0
        )
        
        # Free bits and KL balancing parameters
        self.free_bits = 1.0  # 1 nat ≈ 1.44 bits
        self.kl_alpha = 0.8   # 80% focus on prior learning
        
    def forward(self, obs: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation and hidden state to latent, then get policy logits and value.
        
        Args:
            obs: [batch, obs_dim]
            h_t: [batch, hidden_dim]
            
        Returns:
            logits: [batch, action_dim]
            value: [batch, 1]
            z_t: [batch, latent_dim]
            h_t: [batch, hidden_dim]
        """
        # Encode to latent state
        post_mean, post_std = self.rssm.encode(obs, h_t)
        z_t = self.rssm.sample_latent(post_mean, post_std)
        
        # Get policy logits
        logits = self.policy_net(z_t)
        
        # Get value using DreamerV3 critic
        value = self.value_net(z_t).unsqueeze(-1)  # Add dim for compatibility
        
        return logits, value, z_t, h_t
    
    def act(self, obs: torch.Tensor, h_t: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given observation and hidden state.
        
        Returns:
            action, log_prob, value, z_t, h_t
        """
        logits, value, z_t, h_t = self.forward(obs, h_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob, value, z_t, h_t
    
    def get_value(self, obs: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        _, value, _, _ = self.forward(obs, h_t)
        return value
    
    def predict_reward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict reward using DreamerV3 reward model."""
        return self.reward_net(z_t)
    
    def predict_continue(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict continuation probability."""
        return self.continue_net(z_t)
    
    def update_hidden(self, h_t: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Update hidden state using RSSM transition."""
        h_next, _, _ = self.rssm.transition_step(h_t, z_t, action)
        return h_next
    
    def compute_world_model_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute world model loss with DreamerV3 improvements.
        Includes free bits and KL balancing.
        """
        batch_size, seq_len, _ = obs.shape
        
        # Initialize hidden state
        h_t = self.rssm.h_0.expand(batch_size, -1)
        
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for t in range(seq_len):
            # Encode current observation (posterior)
            post_mean, post_std = self.rssm.encode(obs[:, t], h_t)
            posterior_dist = torch.distributions.Normal(post_mean, post_std)
            
            # Sample from posterior
            z_t = self.rssm.sample_latent(post_mean, post_std)
            
            # Reconstruct observation
            obs_recon = self.rssm.decode(z_t, h_t)
            recon_loss = F.mse_loss(obs_recon, obs[:, t])
            total_recon_loss += recon_loss
            
            # Compute KL loss with prior (if not first timestep)
            if t > 0:
                # Get prior from previous transition
                prior_features = self.rssm.transition.prior_net(h_t)
                prior_mean = self.rssm.transition.prior_mean(prior_features)
                prior_std = F.softplus(self.rssm.transition.prior_std(prior_features)) + 1e-5
                prior_dist = torch.distributions.Normal(prior_mean, prior_std)
                
                # DreamerV3 KL loss with free bits and balancing
                kl_loss = kl_balancing_loss(
                    posterior_dist, prior_dist, 
                    alpha=self.kl_alpha, free_bits=self.free_bits
                )
                total_kl_loss += kl_loss
            
            # Update hidden state for next timestep
            if t < seq_len - 1:
                h_t = self.update_hidden(h_t, z_t, actions[:, t])
        
        # Average losses
        avg_recon_loss = total_recon_loss / seq_len
        avg_kl_loss = total_kl_loss / max(1, seq_len - 1)  # No KL for first timestep
        
        total_loss = avg_recon_loss + avg_kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def compute_reward_model_loss(self, z_batch: torch.Tensor, target_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute reward model loss using DreamerV3 symlog + two-hot encoding.
        """
        return self.reward_net.compute_loss(z_batch, target_rewards)
    
    def compute_value_loss(self, z_batch: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """
        Compute value loss using DreamerV3 symlog + two-hot encoding.
        """
        return self.value_net.compute_loss(z_batch, target_returns)
    
    def imagine_trajectory(self, z_start: torch.Tensor, h_start: torch.Tensor, 
                          steps: int = 15) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory using the learned world model with DreamerV3 improvements.
        """
        batch_size = z_start.shape[0]
        device = z_start.device
        
        # Storage for trajectory
        latent_states = [z_start]
        hidden_states = [h_start]
        actions = []
        log_probs = []
        rewards = []
        values = []
        continues = []
        
        z_t = z_start
        h_t = h_start
        
        for t in range(steps):
            # Sample action from policy
            logits = self.policy_net(z_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value using DreamerV3 critic
            value = self.value_net(z_t)
            
            # Predict reward using DreamerV3 reward model
            reward = self.predict_reward(z_t)
            
            # Predict continue
            continue_prob = self.predict_continue(z_t).squeeze(-1)
            
            # Store
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            continues.append(continue_prob)
            
            # Transition to next state
            action_onehot = torch.zeros(batch_size, self.action_dim, device=device)
            action_onehot.scatter_(1, action.unsqueeze(1), 1)
            
            # Update hidden state
            h_next = self.update_hidden(h_t, z_t, action_onehot)
            
            # Sample next latent state from prior
            prior_features = self.rssm.transition.prior_net(h_next)
            prior_mean = self.rssm.transition.prior_mean(prior_features)
            prior_std = F.softplus(self.rssm.transition.prior_std(prior_features)) + 1e-5
            z_next = self.rssm.sample_latent(prior_mean, prior_std)
            
            # Store states
            latent_states.append(z_next)
            hidden_states.append(h_next)
            
            # Update for next iteration
            z_t = z_next
            h_t = h_next
        
        # Stack all trajectories
        rewards_tensor = torch.stack(rewards)  # [steps, batch_size]
        values_tensor = torch.stack(values)    # [steps, batch_size]
        
        # Update return normalizer with imagined returns (approximate)
        imagined_returns = rewards_tensor.sum(dim=0)  # Simple sum for now
        self.return_normalizer.update(imagined_returns)
        
        return {
            'latent_states': torch.stack(latent_states),    # [steps+1, batch_size, latent_dim]
            'hidden_states': torch.stack(hidden_states),    # [steps+1, batch_size, hidden_dim]
            'actions': torch.stack(actions),                # [steps, batch_size]
            'log_probs': torch.stack(log_probs),            # [steps, batch_size]
            'rewards': rewards_tensor,                      # [steps, batch_size]
            'values': values_tensor,                        # [steps, batch_size]
            'continues': torch.stack(continues)             # [steps, batch_size]
        }
    
    def compute_policy_loss(self, trajectory: Dict[str, torch.Tensor], 
                          gamma: float = 0.99, lambda_: float = 0.95) -> Dict[str, torch.Tensor]:
        """
        Compute policy loss with DreamerV3 return normalization.
        """
        # Extract trajectory components
        rewards = trajectory['rewards']      # [steps, batch_size]
        values = trajectory['values']        # [steps, batch_size]
        continues = trajectory['continues']  # [steps, batch_size]
        log_probs = trajectory['log_probs']  # [steps, batch_size]
        
        steps, batch_size = rewards.shape
        
        # Compute lambda returns (TD(λ))
        returns = torch.zeros_like(rewards)
        
        # Bootstrap from final value
        last_value = values[-1] if steps > 0 else torch.zeros(batch_size, device=rewards.device)
        next_return = last_value
        
        # Compute returns backwards
        for t in reversed(range(steps)):
            if t == steps - 1:
                next_return = last_value
            else:
                next_return = returns[t + 1]
            
            # TD(λ) return
            returns[t] = rewards[t] + gamma * continues[t] * (
                (1 - lambda_) * values[t] + lambda_ * next_return
            )
        
        # Compute advantages
        advantages = returns - values
        
        # DreamerV3: Normalize returns (not advantages)
        # Update normalizer with current returns
        all_returns = returns.flatten()
        self.return_normalizer.update(all_returns)
        
        # Normalize returns
        normalized_returns = self.return_normalizer.normalize(returns)
        normalized_advantages = normalized_returns - values
        
        # Policy loss (REINFORCE with normalized advantages)
        policy_loss = -(log_probs * normalized_advantages.detach()).mean()
        
        # Value loss using DreamerV3 critic
        value_loss = self.value_net.compute_loss(
            trajectory['latent_states'][:-1].view(-1, self.latent_dim),  # Exclude last state
            returns.view(-1)  # Use original returns as targets
        )
        
        # Entropy regularization
        logits = self.policy_net(trajectory['latent_states'][:-1].view(-1, self.latent_dim))
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'mean_return': returns.mean(),
            'mean_advantage': advantages.mean(),
            'return_scale': self.return_normalizer.scale
        }
    
    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Get initial hidden state for RSSM."""
        return self.rssm.h_0.expand(batch_size, -1).to(device) 