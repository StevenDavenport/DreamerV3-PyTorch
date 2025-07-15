#!/usr/bin/env python3
"""
DreamerV3 Training Script with All 4 Key Upgrades:
1. Symlog + Two-hot encoding for reward/value prediction  
2. Percentile-based return normalization
3. Free bits mechanism for KL loss
4. KL balancing for stable world model learning

This should solve the reward prediction issue in sparse reward environments.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.dreamer_v3_agent import DreamerV3Agent
from environments.minigrid_env import MiniGridEnvWrapper
from training.dreamer_buffer import DreamerReplayBuffer

# Weights & Biases for live monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def collect_experience(env, agent, buffer, device, steps=1000):
    """Collect experience from environment."""
    step_count = 0
    episode_count = 0
    episode_rewards = []
    episode_reward = 0
    episode_steps = 0
    
    obs = env.reset()
    h_t = agent.init_hidden(batch_size=1, device=device)
    
    while step_count < steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get action from agent
        action, log_prob, value, z_t, h_t = agent.act(obs_tensor, h_t)
        action_np = action.cpu().numpy()[0]
        
        # Step environment
        next_obs, reward, done, info = env.step(action_np)
        
        # Store transition
        buffer.add(obs, action_np, reward, done, next_obs)
        
        # Update tracking
        episode_reward += reward
        episode_steps += 1
        step_count += 1
        
        # Reset if done
        if done or episode_steps >= 1000:
            episode_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0
            episode_steps = 0
            obs = env.reset()
            h_t = agent.init_hidden(batch_size=1, device=device)
        else:
            obs = next_obs
    
    return len(buffer), episode_rewards

def train_world_model(agent, buffer, device, num_updates=50, batch_size=64):
    """Train world model with DreamerV3 improvements."""
    if not buffer.is_ready(batch_size):
        return {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    
    world_model_optimizer = optim.Adam(agent.rssm.parameters(), lr=3e-4)
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    
    for _ in range(num_updates):
        # Sample sequences for world model training (need temporal sequences)
        seq_len = 16  # Short sequences for stability
        batch = buffer.sample_sequences(batch_size // 4, seq_len)  # Reduce batch size for sequences
        
        obs = batch['observations'].to(device)  # [batch, seq_len, obs_dim]
        actions = batch['actions'].to(device)   # [batch, seq_len, action_dim]
        
        # Compute world model loss with DreamerV3 improvements
        loss_dict = agent.compute_world_model_loss(obs, actions)
        
        # Backward pass
        world_model_optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(agent.rssm.parameters(), 1.0)
        world_model_optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key].item()
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_updates
    
    return total_losses

def train_reward_model(agent, buffer, device, num_updates=100, batch_size=64):
    """Train reward model with DreamerV3 symlog + two-hot encoding."""
    if not buffer.is_ready(batch_size):
        return 0.0
    
    reward_optimizer = optim.Adam(agent.reward_net.parameters(), lr=3e-4)
    total_loss = 0.0
    
    for _ in range(num_updates):
        # Sample batch
        batch = buffer.sample_batch(batch_size)
        
        obs = batch['observations'].to(device)
        rewards = batch['rewards'].to(device)
        
        # Get latent states
        h_batch = agent.init_hidden(batch_size=batch_size, device=device)
        z_means, z_stds = agent.rssm.encode(obs, h_batch)
        z_batch = agent.rssm.sample_latent(z_means, z_stds)
        
        # Compute reward loss using DreamerV3 improvements
        loss = agent.compute_reward_model_loss(z_batch, rewards)
        
        # Backward pass
        reward_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.reward_net.parameters(), 1.0)
        reward_optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_updates

def train_continue_model(agent, buffer, device, num_updates=50, batch_size=64):
    """Train continue model (unchanged for now)."""
    if not buffer.is_ready(batch_size):
        return 0.0
    
    continue_optimizer = optim.Adam(agent.continue_net.parameters(), lr=3e-4)
    total_loss = 0.0
    
    for _ in range(num_updates):
        # Sample batch
        batch = buffer.sample_batch(batch_size)
        
        obs = batch['observations'].to(device)
        dones = batch['dones'].to(device)
        
        # Get latent states
        h_batch = agent.init_hidden(batch_size=batch_size, device=device)
        z_means, z_stds = agent.rssm.encode(obs, h_batch)
        z_batch = agent.rssm.sample_latent(z_means, z_stds)
        
        # Predict continue (1 - done)
        pred_continues = agent.predict_continue(z_batch).squeeze(-1)
        target_continues = 1.0 - dones.float()
        
        # Compute loss
        loss = nn.BCELoss()(pred_continues, target_continues)
        
        # Backward pass
        continue_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.continue_net.parameters(), 1.0)
        continue_optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_updates

def train_policy_value(agent, buffer, device, num_updates=50, imagination_horizon=15):
    """Train policy and value using imagination with DreamerV3 improvements."""
    if not buffer.is_ready(32):
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    
    policy_optimizer = optim.Adam([
        {'params': agent.policy_net.parameters(), 'lr': 3e-4},
        {'params': agent.value_net.parameters(), 'lr': 3e-4}
    ])
    
    total_losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'mean_return': 0.0}
    
    for _ in range(num_updates):
        # Sample starting states from buffer
        batch = buffer.sample_batch(32)
        obs = batch['observations'].to(device)
        
        # Get starting latent states
        h_start = agent.init_hidden(batch_size=32, device=device)
        z_means, z_stds = agent.rssm.encode(obs, h_start)
        z_start = agent.rssm.sample_latent(z_means, z_stds)
        
        # Imagine trajectory using DreamerV3 agent
        trajectory = agent.imagine_trajectory(z_start, h_start, steps=imagination_horizon)
        
        # Compute policy loss with DreamerV3 return normalization
        loss_dict = agent.compute_policy_loss(trajectory)
        
        # Total loss
        total_loss = (loss_dict['policy_loss'] + 
                     loss_dict['value_loss'] - 
                     0.01 * loss_dict['entropy'])  # Entropy regularization
        
        # Backward pass
        policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.policy_net.parameters()) + 
                                     list(agent.value_net.parameters()), 1.0)
        policy_optimizer.step()
        
        # Accumulate losses
        for key in ['policy_loss', 'value_loss', 'entropy', 'mean_return']:
            total_losses[key] += loss_dict[key].item()
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_updates
    
    return total_losses

def evaluate_agent(env, agent, device, num_episodes=10):
    """Evaluate agent performance."""
    episode_rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        h_t = agent.init_hidden(batch_size=1, device=device)
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 1000:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _, h_t = agent.act(obs_tensor, h_t, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, done, info = env.step(action_np)
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                if reward > 0.5:  # Success in MiniGrid
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
    
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    return avg_reward, success_rate

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="dreamer-v3-minigrid",
            name="dreamerv3-empty-6x6-all-upgrades",
            config={
                "environment": "MiniGrid-Empty-6x6-v0",
                "algorithm": "DreamerV3",
                "device": device,
                "upgrades": ["symlog_two_hot", "return_normalization", "free_bits", "kl_balancing"],
                "rssm_hidden_dim": 128,
                "rssm_latent_dim": 64,
                "policy_hidden_dim": 256,
                "value_hidden_dim": 256,
                "imagination_horizon": 15,
                "buffer_capacity": 50000,
                "collect_steps": 1000,
                "world_model_updates": 50,
                "reward_model_updates": 100,
                "policy_updates": 50,
            }
        )
    
    # Environment
    env = MiniGridEnvWrapper(env_name="MiniGrid-Empty-6x6-v0")
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()
    print(f"MiniGrid-Empty-6x6 obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    # DreamerV3 Agent with all upgrades
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        rssm_hidden_dim=128,
        rssm_latent_dim=64,
        policy_hidden_dim=256,
        value_hidden_dim=256,
        reward_hidden_dim=256,
        continue_hidden_dim=256,
        num_reward_bins=41,
        num_value_bins=41
    ).to(device)
    
    # Replay buffer
    buffer = DreamerReplayBuffer(capacity=50000, obs_dim=obs_dim, action_dim=action_dim)
    
    # Training parameters
    num_iterations = 200
    collect_steps = 1000
    world_model_updates = 50
    reward_model_updates = 100  # More updates for reward model
    continue_model_updates = 50
    policy_updates = 50
    eval_frequency = 20
    
    # Logging
    metrics = {
        'episode_rewards': [],
        'success_rates': [],
        'world_model_losses': [],
        'reward_model_losses': [],
        'continue_model_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'return_scales': []
    }
    
    print("Starting DreamerV3 training with all 4 upgrades...")
    print("Expected to solve the sparse reward issue!")
    
    for iteration in range(num_iterations):
        # 1. Collect real environment experience
        buffer_size, episode_rewards = collect_experience(env, agent, buffer, device, collect_steps)
        
        if len(episode_rewards) > 0:
            recent_reward = np.mean(episode_rewards)
            print(f"Iteration {iteration}: Recent episode reward: {recent_reward:.3f}")
        
        if not buffer.is_ready(1000):
            continue
        
        # 2. Train world model with DreamerV3 improvements
        wm_losses = train_world_model(agent, buffer, device, world_model_updates)
        metrics['world_model_losses'].append(wm_losses['total_loss'])
        
        # 3. Train reward model with symlog + two-hot encoding
        rm_loss = train_reward_model(agent, buffer, device, reward_model_updates)
        metrics['reward_model_losses'].append(rm_loss)
        
        # 4. Train continue model
        cm_loss = train_continue_model(agent, buffer, device, continue_model_updates)
        metrics['continue_model_losses'].append(cm_loss)
        
        # 5. Train policy and value using imagination with return normalization
        pv_losses = train_policy_value(agent, buffer, device, policy_updates)
        metrics['policy_losses'].append(pv_losses['policy_loss'])
        metrics['value_losses'].append(pv_losses['value_loss'])
        metrics['return_scales'].append(agent.return_normalizer.scale)
        
        # 6. Evaluate agent
        if iteration % eval_frequency == 0:
            avg_reward, success_rate = evaluate_agent(env, agent, device)
            metrics['episode_rewards'].append(avg_reward)
            metrics['success_rates'].append(success_rate)
            
            print(f"Iteration {iteration}:")
            print(f"  Buffer size: {buffer_size}")
            print(f"  World Model - Total: {wm_losses['total_loss']:.4f}, Recon: {wm_losses['recon_loss']:.4f}, KL: {wm_losses['kl_loss']:.4f}")
            print(f"  Reward Model: {rm_loss:.4f}")
            print(f"  Continue Model: {cm_loss:.4f}")
            print(f"  Policy: {pv_losses['policy_loss']:.4f}, Value: {pv_losses['value_loss']:.4f}")
            print(f"  Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.1%}")
            print(f"  Return Scale: {agent.return_normalizer.scale:.3f}")
            print(f"  Mean Imagined Return: {pv_losses['mean_return']:.3f}")
            
            # Log to wandb
            if WANDB_AVAILABLE:
                wandb.log({
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'world_model_loss': wm_losses['total_loss'],
                    'recon_loss': wm_losses['recon_loss'],
                    'kl_loss': wm_losses['kl_loss'],
                    'reward_model_loss': rm_loss,
                    'continue_model_loss': cm_loss,
                    'policy_loss': pv_losses['policy_loss'],
                    'value_loss': pv_losses['value_loss'],
                    'entropy': pv_losses['entropy'],
                    'return_scale': agent.return_normalizer.scale,
                    'mean_imagined_return': pv_losses['mean_return'],
                    'buffer_size': buffer_size
                })
    
    # Plot results
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 4, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Average Episode Reward')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    
    plt.subplot(2, 4, 2)
    plt.plot(metrics['success_rates'])
    plt.title('Success Rate')
    plt.xlabel('Evaluation')
    plt.ylabel('Success Rate')
    
    plt.subplot(2, 4, 3)
    plt.plot(metrics['world_model_losses'])
    plt.title('World Model Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 4, 4)
    plt.plot(metrics['reward_model_losses'])
    plt.title('DreamerV3 Reward Model Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 4, 5)
    plt.plot(metrics['continue_model_losses'])
    plt.title('Continue Model Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 4, 6)
    plt.plot(metrics['policy_losses'])
    plt.title('Policy Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 4, 7)
    plt.plot(metrics['value_losses'])
    plt.title('DreamerV3 Value Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 4, 8)
    plt.plot(metrics['return_scales'])
    plt.title('Return Normalization Scale')
    plt.xlabel('Iteration')
    plt.ylabel('Scale')
    
    plt.tight_layout()
    plt.savefig('dreamer_v3_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final evaluation
    final_reward, final_success = evaluate_agent(env, agent, device, num_episodes=50)
    print(f"\nFinal Results (50 episodes):")
    print(f"Average Reward: {final_reward:.3f}")
    print(f"Success Rate: {final_success:.1%}")
    
    # Test reward model quality
    print(f"\nDreamerV3 Improvements Summary:")
    print(f"✅ Symlog + Two-hot encoding: Handles reward scale imbalance")
    print(f"✅ Return normalization scale: {agent.return_normalizer.scale:.3f}")
    print(f"✅ Free bits KL loss: Prevents degenerate solutions")
    print(f"✅ KL balancing: Stable world model learning")
    
    if final_success > 0.5:
        print(f"\n🎉 SUCCESS! DreamerV3 upgrades solved the sparse reward issue!")
        print(f"   Achieved {final_success:.1%} success rate vs ~0% with original Dreamer")
    else:
        print(f"\n⚠️  Still need more improvements or longer training...")

if __name__ == "__main__":
    train() 