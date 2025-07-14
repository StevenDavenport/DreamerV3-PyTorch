import torch
import torch.nn as nn
import torch.nn.functional as F
from models.world_model import RSSMWorldModel
from typing import Dict

class RSSMBaseAgent(nn.Module):
    """
    Base agent using RSSM for latent state, with actor-critic heads, reward model, and continue model.
    Suitable for Dreamer-style imagination-based RL.
    """
    def __init__(self, obs_dim, action_dim, rssm_hidden_dim, rssm_latent_dim, policy_hidden_dim=256, value_hidden_dim=256, reward_hidden_dim=256, continue_hidden_dim=256):
        super().__init__()
        self.rssm = RSSMWorldModel(obs_dim, action_dim, rssm_hidden_dim, rssm_latent_dim)
        self.action_dim = action_dim
        self.latent_dim = rssm_latent_dim
        self.hidden_dim = rssm_hidden_dim

        # Policy (actor) head
        self.policy_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim)
        )
        # Value (critic) head
        self.value_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
        # Reward model for imagination
        self.reward_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, reward_hidden_dim),
            nn.ReLU(),
            nn.Linear(reward_hidden_dim, 1)
        )
        # Continue model for imagination (predicts 1-done)
        self.continue_net = nn.Sequential(
            nn.Linear(rssm_latent_dim, continue_hidden_dim),
            nn.ReLU(),
            nn.Linear(continue_hidden_dim, 1),
            nn.Sigmoid()  # Output probability of continuing
        )

    def forward(self, obs, h_t):
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
        post_mean, post_std = self.rssm.encode(obs, h_t)
        z_t = self.rssm.sample_latent(post_mean, post_std)
        logits = self.policy_net(z_t)
        value = self.value_net(z_t)
        return logits, value, z_t, h_t

    def act(self, obs, h_t, deterministic=False):
        """
        Select action given observation and hidden state.
        Returns action, log_prob, value, next latent, next hidden.
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

    def get_value(self, obs, h_t):
        """
        Get value estimate for given observation and hidden state.
        """
        _, value, _, _ = self.forward(obs, h_t)
        return value

    def predict_reward(self, z_t):
        """
        Predict reward from latent state.
        """
        return self.reward_net(z_t)

    def predict_continue(self, z_t):
        """
        Predict continuation probability from latent state.
        Returns probability of episode continuing (1 - done probability).
        """
        return self.continue_net(z_t)

    def update_hidden(self, h_t, z_t, action):
        """
        Update hidden state using RSSM transition.
        Args:
            h_t: [batch, hidden_dim]
            z_t: [batch, latent_dim]
            action: [batch, action_dim] (one-hot)
        Returns:
            h_next: [batch, hidden_dim]
        """
        h_next, _, _ = self.rssm.transition_step(h_t, z_t, action)
        return h_next

    def imagine_trajectory(self, z_start: torch.Tensor, h_start: torch.Tensor, 
                          steps: int = 15) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory using the learned world model.
        
        Args:
            z_start: Starting latent state [batch_size, latent_dim]
            h_start: Starting hidden state [batch_size, hidden_dim]
            steps: Number of imagination steps
            
        Returns:
            Dictionary containing imagined trajectory
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
            
            # Get value
            value = self.value_net(z_t).squeeze(-1)
            
            # Predict reward
            reward = self.predict_reward(z_t).squeeze(-1)
            
            # Predict continue (1 - done)
            continue_logit = self.predict_continue(z_t).squeeze(-1)
            continue_prob = torch.sigmoid(continue_logit)
            
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
            prior_std = torch.nn.functional.softplus(
                self.rssm.transition.prior_std(prior_features)
            ) + 1e-5
            z_next = self.rssm.sample_latent(prior_mean, prior_std)
            
            # Store states
            latent_states.append(z_next)
            hidden_states.append(h_next)
            
            # Update for next iteration
            z_t = z_next
            h_t = h_next
        
        return {
            'latent_states': torch.stack(latent_states),  # [steps+1, batch_size, latent_dim]
            'hidden_states': torch.stack(hidden_states),  # [steps+1, batch_size, hidden_dim]
            'actions': torch.stack(actions),              # [steps, batch_size]
            'log_probs': torch.stack(log_probs),          # [steps, batch_size]
            'rewards': torch.stack(rewards),              # [steps, batch_size]
            'values': torch.stack(values),                # [steps, batch_size]
            'continues': torch.stack(continues)           # [steps, batch_size]
        }

    def init_hidden(self, batch_size=1, device="cpu"):
        """
        Get initial hidden state for RSSM.
        """
        return self.rssm.h_0.expand(batch_size, -1).to(device) 