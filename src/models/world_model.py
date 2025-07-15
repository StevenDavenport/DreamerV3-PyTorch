"""
RSSM-style world model for skill-conditioned MBRL.
Implements a recurrent state space model with encoder, decoder, and transition dynamics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class RSSMEncoder(nn.Module):
    """Encoder that maps observations to latent representations."""
    
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Posterior network (q(z_t | o_t, h_t))
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_std = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation and hidden state to posterior distribution.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            h_t: Hidden state tensor [batch_size, hidden_dim]
            
        Returns:
            posterior_mean: Mean of posterior distribution
            posterior_std: Standard deviation of posterior distribution
        """
        # Encode observation
        obs_encoded = self.obs_encoder(obs)
        
        # Combine with hidden state
        combined = torch.cat([obs_encoded, h_t], dim=-1)
        posterior_features = self.posterior_net(combined)
        
        # Output distribution parameters
        posterior_mean = self.posterior_mean(posterior_features)
        posterior_std = F.softplus(self.posterior_std(posterior_features)) + 1e-5
        
        return posterior_mean, posterior_std


class RSSMTransition(nn.Module):
    """Transition model that predicts next latent state given current state and action."""
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Prior network (p(z_t | h_t))
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)
        
        # Transition network (p(h_{t+1} | h_t, z_t, a_t))
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next hidden state and prior distribution.
        
        Args:
            h_t: Current hidden state [batch_size, hidden_dim]
            z_t: Current latent state [batch_size, latent_dim]
            a_t: Current action [batch_size, action_dim]
            
        Returns:
            h_next: Next hidden state
            prior_mean: Mean of prior distribution
            prior_std: Standard deviation of prior distribution
        """
        # Compute prior distribution
        prior_features = self.prior_net(h_t)
        prior_mean = self.prior_mean(prior_features)
        prior_std = F.softplus(self.prior_std(prior_features)) + 1e-5
        
        # Compute next hidden state
        transition_input = torch.cat([h_t, z_t, a_t], dim=-1)
        h_next = self.transition_net(transition_input)
        
        return h_next, prior_mean, prior_std


class RSSMDecoder(nn.Module):
    """Decoder that reconstructs observations from latent states."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state and hidden state to observation.
        
        Args:
            z_t: Latent state [batch_size, latent_dim]
            h_t: Hidden state [batch_size, hidden_dim]
            
        Returns:
            obs_recon: Reconstructed observation
        """
        combined = torch.cat([z_t, h_t], dim=-1)
        obs_recon = self.decoder(combined)
        return obs_recon


class RSSMWorldModel(nn.Module):
    """Complete RSSM world model combining encoder, transition, and decoder."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.encoder = RSSMEncoder(obs_dim, hidden_dim, latent_dim)
        self.transition = RSSMTransition(latent_dim, action_dim, hidden_dim)
        self.decoder = RSSMDecoder(latent_dim, hidden_dim, obs_dim)
        
        # Initialize hidden state
        self.h_0 = nn.Parameter(torch.zeros(1, hidden_dim))
        
    def encode(self, obs: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to posterior distribution."""
        return self.encoder(obs, h_t)
    
    def transition_step(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one transition step."""
        return self.transition(h_t, z_t, a_t)
    
    def decode(self, z_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation."""
        return self.decoder(z_t, h_t)
    
    def sample_latent(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian distribution."""
        return mean + std * torch.randn_like(std)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the world model.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            actions: Actions [batch_size, seq_len, action_dim]
            
        Returns:
            Dictionary containing all intermediate states and reconstructions
        """
        batch_size, seq_len, _ = obs.shape
        
        # Initialize states
        h_t = self.h_0.expand(batch_size, -1)
        
        # Storage for outputs
        posterior_means = []
        posterior_stds = []
        prior_means = []
        prior_stds = []
        latent_samples = []
        hidden_states = []
        obs_reconstructions = []
        
        for t in range(seq_len):
            # Encode current observation
            post_mean, post_std = self.encode(obs[:, t], h_t)
            posterior_means.append(post_mean)
            posterior_stds.append(post_std)
            
            # Sample from posterior
            z_t = self.sample_latent(post_mean, post_std)
            latent_samples.append(z_t)
            
            # Decode observation
            obs_recon = self.decode(z_t, h_t)
            obs_reconstructions.append(obs_recon)
            
            # Store hidden state
            hidden_states.append(h_t)
            
            # Predict next hidden state and prior (for next step)
            if t < seq_len - 1:
                h_next, prior_mean, prior_std = self.transition_step(h_t, z_t, actions[:, t])
                prior_means.append(prior_mean)
                prior_stds.append(prior_std)
                h_t = h_next
        
        return {
            'posterior_means': torch.stack(posterior_means, dim=1),
            'posterior_stds': torch.stack(posterior_stds, dim=1),
            'prior_means': torch.stack(prior_means, dim=1) if prior_means else None,
            'prior_stds': torch.stack(prior_stds, dim=1) if prior_stds else None,
            'latent_samples': torch.stack(latent_samples, dim=1),
            'hidden_states': torch.stack(hidden_states, dim=1),
            'obs_reconstructions': torch.stack(obs_reconstructions, dim=1)
        }
    
    def imagine_trajectory(self, z_0: torch.Tensor, h_0: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory using the transition model.
        
        Args:
            z_0: Initial latent state [batch_size, latent_dim]
            h_0: Initial hidden state [batch_size, hidden_dim]
            actions: Actions to take [batch_size, seq_len, action_dim]
            
        Returns:
            Dictionary containing imagined trajectory
        """
        batch_size, seq_len, _ = actions.shape
        
        # Storage for outputs
        latent_states = [z_0]
        hidden_states = [h_0]
        obs_reconstructions = []
        
        h_t = h_0
        z_t = z_0
        
        for t in range(seq_len):
            # Decode current state
            obs_recon = self.decode(z_t, h_t)
            obs_reconstructions.append(obs_recon)
            
            # Transition to next state
            h_next, _, _ = self.transition_step(h_t, z_t, actions[:, t])
            
            # Sample next latent state from prior
            prior_features = self.transition.prior_net(h_next)
            prior_mean = self.transition.prior_mean(prior_features)
            prior_std = F.softplus(self.transition.prior_std(prior_features)) + 1e-5
            z_next = self.sample_latent(prior_mean, prior_std)
            
            # Update states
            h_t = h_next
            z_t = z_next
            
            # Store states
            latent_states.append(z_t)
            hidden_states.append(h_t)
        
        return {
            'latent_states': torch.stack(latent_states, dim=1),
            'hidden_states': torch.stack(hidden_states, dim=1),
            'obs_reconstructions': torch.stack(obs_reconstructions, dim=1)
        }


def compute_rssm_loss(world_model: RSSMWorldModel, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute RSSM loss components.
    
    Args:
        world_model: RSSM world model
        obs: Observations [batch_size, seq_len, obs_dim]
        actions: Actions [batch_size, seq_len, action_dim]
        
    Returns:
        Dictionary containing loss components
    """
    # Forward pass
    outputs = world_model(obs, actions)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(outputs['obs_reconstructions'], obs)
    
    # KL divergence loss (if we have prior)
    kl_loss = 0.0
    if outputs['prior_means'] is not None:
        # Compute KL divergence between posterior and prior
        posterior_dist = torch.distributions.Normal(outputs['posterior_means'][:, :-1], outputs['posterior_stds'][:, :-1])
        prior_dist = torch.distributions.Normal(outputs['prior_means'], outputs['prior_stds'])
        kl_loss = torch.mean(
            torch.sum(
                torch.distributions.kl_divergence(posterior_dist, prior_dist),
                dim=-1
            )
        )
    
    # Total loss
    total_loss = recon_loss + kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    } 