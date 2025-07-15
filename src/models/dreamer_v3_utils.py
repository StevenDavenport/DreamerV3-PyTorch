"""
DreamerV3 utility functions for robust reward prediction and training.
Includes symlog transformations, two-hot encoding, and return normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm transformation from DreamerV3.
    Compresses magnitudes of large positive and negative values.
    Approximates identity around origin.
    
    Formula: symlog(x) = sign(x) * log(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog transformation.
    
    Formula: symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def create_exponential_bins(num_bins: int = 41, min_val: float = -20.0, max_val: float = 20.0) -> torch.Tensor:
    """
    Create exponentially spaced bins for two-hot encoding.
    
    Args:
        num_bins: Number of bins (default 41 for range [-20, 20])
        min_val: Minimum value in symlog space
        max_val: Maximum value in symlog space
        
    Returns:
        bins: Tensor of bin locations in original space
    """
    # Create linear spacing in symlog space
    symlog_bins = torch.linspace(min_val, max_val, num_bins)
    # Transform back to original space
    bins = symexp(symlog_bins)
    return bins


def two_hot_encode(targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Two-hot encoding for continuous targets.
    Generalizes one-hot encoding by distributing weight between the two closest bins.
    
    Args:
        targets: Target values to encode [batch_size, ...]
        bins: Bin locations [num_bins]
        
    Returns:
        encoding: Two-hot encoded targets [batch_size, ..., num_bins]
    """
    # Flatten targets for processing
    original_shape = targets.shape
    targets_flat = targets.flatten()
    
    # Find closest bin indices
    distances = torch.abs(targets_flat.unsqueeze(-1) - bins.unsqueeze(0))  # [batch_flat, num_bins]
    closest_indices = torch.argmin(distances, dim=-1)  # [batch_flat]
    
    # Get the two closest bins
    num_bins = bins.shape[0]
    batch_size = targets_flat.shape[0]
    
    # Initialize encoding
    encoding = torch.zeros(batch_size, num_bins, device=targets.device, dtype=targets.dtype)
    
    for i in range(batch_size):
        target_val = targets_flat[i]
        closest_idx = closest_indices[i].item()
        
        # Handle edge cases
        if closest_idx == 0:
            # Target is closest to first bin
            next_idx = 1
        elif closest_idx == num_bins - 1:
            # Target is closest to last bin
            next_idx = closest_idx - 1
            closest_idx = next_idx
            next_idx = num_bins - 1
        else:
            # Check which neighbor is closer
            left_dist = torch.abs(target_val - bins[closest_idx - 1])
            right_dist = torch.abs(target_val - bins[closest_idx + 1])
            
            if left_dist < right_dist:
                next_idx = closest_idx - 1
                closest_idx = next_idx
                next_idx = closest_indices[i].item()
            else:
                next_idx = closest_idx + 1
        
        # Compute weights
        bin1_val = bins[closest_idx]
        bin2_val = bins[next_idx]
        
        if torch.abs(bin2_val - bin1_val) < 1e-8:
            # Bins are too close, use equal weights
            weight1 = 0.5
            weight2 = 0.5
        else:
            distance_total = torch.abs(bin2_val - bin1_val)
            distance_to_bin2 = torch.abs(target_val - bin2_val)
            weight1 = distance_to_bin2 / distance_total
            weight2 = 1.0 - weight1
        
        encoding[i, closest_idx] = weight1
        encoding[i, next_idx] = weight2
    
    # Reshape back to original shape + bins dimension
    encoding = encoding.view(*original_shape, num_bins)
    return encoding


def two_hot_decode(encoding: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Decode two-hot encoding back to continuous values.
    
    Args:
        encoding: Two-hot encoded values [..., num_bins]
        bins: Bin locations [num_bins]
        
    Returns:
        decoded: Continuous values [...]
    """
    # Weighted average of bin locations
    decoded = torch.sum(encoding * bins.unsqueeze(0), dim=-1)
    return decoded


class SymlogTwoHotLoss(nn.Module):
    """
    Loss function using symlog transformation and two-hot encoding.
    Robust to different reward scales and handles multi-modal distributions.
    """
    
    def __init__(self, num_bins: int = 41, min_val: float = -20.0, max_val: float = 20.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        
        # Create bins (will be moved to device when first used)
        self.register_buffer('bins', create_exponential_bins(num_bins, min_val, max_val))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute symlog two-hot loss.
        
        Args:
            logits: Network outputs [batch_size, ..., num_bins]
            targets: Target values [batch_size, ...]
            
        Returns:
            loss: Cross-entropy loss
        """
        # Transform targets to symlog space
        targets_symlog = symlog(targets)
        
        # Two-hot encode targets in symlog space
        targets_encoding = two_hot_encode(targets_symlog, 
                                        torch.linspace(self.min_val, self.max_val, self.num_bins, 
                                                     device=targets.device))
        
        # Compute cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(targets_encoding * log_probs, dim=-1)
        return loss.mean()


class SymlogTwoHotRewardPredictor(nn.Module):
    """
    Reward predictor using symlog transformation and two-hot encoding.
    Replaces simple regression for robust reward prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_bins: int = 41):
        super().__init__()
        self.num_bins = num_bins
        
        # Network outputs logits for categorical distribution
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )
        
        # Loss function
        self.loss_fn = SymlogTwoHotLoss(num_bins)
        
        # Create bins for decoding
        self.register_buffer('bins', create_exponential_bins(num_bins))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning continuous reward predictions.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            rewards: Predicted rewards [batch_size]
        """
        logits = self.network(x)
        probs = F.softmax(logits, dim=-1)
        
        # Decode from two-hot to continuous values
        symlog_values = torch.sum(probs * torch.linspace(self.loss_fn.min_val, self.loss_fn.max_val, 
                                                        self.num_bins, device=x.device), dim=-1)
        rewards = symexp(symlog_values)
        return rewards
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            x: Input features [batch_size, input_dim]
            targets: Target rewards [batch_size]
            
        Returns:
            loss: Training loss
        """
        logits = self.network(x)
        return self.loss_fn(logits, targets)


class SymlogTwoHotCritic(nn.Module):
    """
    Value function critic using symlog transformation and two-hot encoding.
    Handles multi-modal return distributions and extreme values.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_bins: int = 41):
        super().__init__()
        self.num_bins = num_bins
        
        # Network outputs logits for categorical distribution
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )
        
        # Loss function
        self.loss_fn = SymlogTwoHotLoss(num_bins)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning continuous value predictions.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            values: Predicted values [batch_size]
        """
        logits = self.network(x)
        probs = F.softmax(logits, dim=-1)
        
        # Decode from two-hot to continuous values
        symlog_values = torch.sum(probs * torch.linspace(self.loss_fn.min_val, self.loss_fn.max_val, 
                                                        self.num_bins, device=x.device), dim=-1)
        values = symexp(symlog_values)
        return values
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            x: Input features [batch_size, input_dim]
            targets: Target returns [batch_size]
            
        Returns:
            loss: Training loss
        """
        logits = self.network(x)
        return self.loss_fn(logits, targets)


class ReturnNormalizer:
    """
    Percentile-based return normalization from DreamerV3.
    Handles outliers and sparse rewards robustly.
    """
    
    def __init__(self, percentile_low: float = 5.0, percentile_high: float = 95.0, 
                 ema_alpha: float = 0.99, min_threshold: float = 1.0):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        
        # Running statistics
        self.scale = 1.0
        self.initialized = False
        
    def update(self, returns: torch.Tensor):
        """
        Update normalization statistics with new returns.
        
        Args:
            returns: Batch of returns [batch_size]
        """
        if returns.numel() == 0:
            return
            
        # Compute percentiles
        low_val = torch.quantile(returns, self.percentile_low / 100.0)
        high_val = torch.quantile(returns, self.percentile_high / 100.0)
        
        # Compute range
        range_val = (high_val - low_val).item()
        
        if not self.initialized:
            self.scale = max(range_val, 1.0)
            self.initialized = True
        else:
            # Exponential moving average
            self.scale = self.ema_alpha * self.scale + (1 - self.ema_alpha) * max(range_val, 1.0)
    
    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Normalize returns using current statistics.
        Only scales down large returns, preserves small ones.
        
        Args:
            returns: Returns to normalize [batch_size]
            
        Returns:
            normalized: Normalized returns [batch_size]
        """
        if not self.initialized:
            return returns
            
        # Only scale down returns larger than threshold
        denominator = max(self.scale, self.min_threshold)
        return returns / denominator
    
    def __call__(self, returns: torch.Tensor) -> torch.Tensor:
        """Convenience method for normalization."""
        return self.normalize(returns)


def free_bits_kl_loss(posterior_dist: torch.distributions.Distribution,
                     prior_dist: torch.distributions.Distribution,
                     free_bits: float = 1.0) -> torch.Tensor:
    """
    KL divergence loss with free bits mechanism.
    Clips KL loss below threshold to focus learning on prediction.
    
    Args:
        posterior_dist: Posterior distribution
        prior_dist: Prior distribution  
        free_bits: Free bits threshold (1.0 nat ≈ 1.44 bits)
        
    Returns:
        loss: Clipped KL divergence loss
    """
    kl_div = torch.distributions.kl_divergence(posterior_dist, prior_dist)
    # Clip below free_bits threshold
    kl_loss = torch.maximum(kl_div, torch.tensor(free_bits, device=kl_div.device))
    return kl_loss.mean()


def kl_balancing_loss(posterior_dist: torch.distributions.Distribution,
                     prior_dist: torch.distributions.Distribution,
                     alpha: float = 0.8,
                     free_bits: float = 1.0) -> torch.Tensor:
    """
    KL loss with balancing between prior and posterior learning.
    Uses different learning rates for prior vs posterior (α for prior, 1-α for posterior).
    
    Args:
        posterior_dist: Posterior distribution
        prior_dist: Prior distribution
        alpha: Weight for prior learning (0.8 means 80% focus on prior)
        free_bits: Free bits threshold
        
    Returns:
        loss: Balanced KL divergence loss
    """
    # Create stop-gradient versions
    posterior_sg = torch.distributions.Normal(posterior_dist.mean.detach(), 
                                            posterior_dist.stddev.detach())
    prior_sg = torch.distributions.Normal(prior_dist.mean.detach(), 
                                        prior_dist.stddev.detach())
    
    # Balanced KL loss
    kl_loss = (alpha * torch.distributions.kl_divergence(posterior_sg, prior_dist) + 
              (1 - alpha) * torch.distributions.kl_divergence(posterior_dist, prior_sg))
    
    # Apply free bits
    kl_loss = torch.maximum(kl_loss, torch.tensor(free_bits, device=kl_loss.device))
    return kl_loss.mean() 