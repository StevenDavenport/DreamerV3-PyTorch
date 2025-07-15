#!/usr/bin/env python3
"""
DreamerV3 Weekend Challenge: MiniGrid-FourRooms-v0

A much harder environment to test the robustness of our DreamerV3 implementation:
- Sparse rewards (only at goal)
- Long episodes (20-50+ steps)  
- Complex navigation (4 rooms, 3 doorways)
- Requires planning and memory

Perfect test for imagination-based RL!
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

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

class WeekendCheckpointer:
    """Extended checkpointer for weekend training."""
    
    def __init__(self, save_dir: str = "weekend_experiments", patience: int = 50):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"dreamerv3_fourrooms_weekend_{timestamp}"
        
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        
        # Performance tracking
        self.best_success_rate = -1.0
        self.best_avg_reward = -float('inf')
        self.iterations_without_improvement = 0
        self.best_iteration = -1
        
        print(f"🏠 Weekend Experiment: {experiment_name}")
        print(f"📁 Saving to: {self.save_dir}")
        print(f"🛡️ Early stopping patience: {self.patience} iterations")
        print(f"⏰ Extended training for weekend run!")
    
    def should_save(self, success_rate: float, avg_reward: float) -> bool:
        """Determine if current model is better than previous best."""
        if success_rate > self.best_success_rate:
            return True
        elif success_rate == self.best_success_rate and avg_reward > self.best_avg_reward:
            return True
        return False
    
    def save_checkpoint(self, agent: DreamerV3Agent, iteration: int, 
                       success_rate: float, avg_reward: float, 
                       metrics: dict) -> bool:
        """Save model checkpoint if performance improved."""
        
        improved = self.should_save(success_rate, avg_reward)
        
        if improved:
            # Save new best model
            self.best_success_rate = success_rate
            self.best_avg_reward = avg_reward
            self.best_iteration = iteration
            self.iterations_without_improvement = 0
            
            # Create descriptive model name
            model_name = f"dreamerv3_fourrooms_iter{iteration:03d}_success{success_rate:.0%}_reward{avg_reward:.3f}"
            
            checkpoint = {
                'iteration': iteration,
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'agent_state_dict': agent.state_dict(),
                'return_normalizer_scale': agent.return_normalizer.scale,
                'return_normalizer_initialized': agent.return_normalizer.initialized,
                'metrics': metrics,
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'environment': 'MiniGrid-FourRooms-v0'
            }
            
            # Save with descriptive name
            checkpoint_path = self.save_dir / f"{model_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Also save as "best_model.pt" for easy loading
            best_model_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            
            # Save simple progress log
            self._save_progress_log(iteration, success_rate, avg_reward, metrics)
            
            print(f"💾 NEW BEST MODEL SAVED!")
            print(f"   Iteration: {iteration}")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Reward: {avg_reward:.3f}")
            print(f"   Model: {model_name}")
            
            return True
        else:
            self.iterations_without_improvement += 1
            if self.iterations_without_improvement % 10 == 0:  # Less frequent logging
                print(f"⏳ No improvement for {self.iterations_without_improvement}/{self.patience} iterations")
            return False
    
    def should_stop(self) -> bool:
        """Check if training should stop due to lack of improvement."""
        return self.iterations_without_improvement >= self.patience
    
    def _save_progress_log(self, iteration: int, success_rate: float, avg_reward: float, metrics: dict):
        """Save progress log for weekend monitoring."""
        log_path = self.save_dir / "weekend_progress.txt"
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ")
            f.write(f"Iter {iteration:03d} | ")
            f.write(f"Success: {success_rate:.1%} | ")
            f.write(f"Reward: {avg_reward:.3f} | ")
            f.write(f"NEW BEST!\n")
    
    def load_best_model(self, agent: DreamerV3Agent) -> bool:
        """Load the best saved model."""
        checkpoint_path = self.save_dir / "best_model.pt"
        
        if not checkpoint_path.exists():
            print(f"❌ No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            agent.return_normalizer.scale = checkpoint['return_normalizer_scale']
            agent.return_normalizer.initialized = checkpoint['return_normalizer_initialized']
            
            print(f"✅ Loaded best weekend model:")
            print(f"   Iteration: {checkpoint['iteration']}")
            print(f"   Success Rate: {checkpoint['success_rate']:.1%}")
            print(f"   Avg Reward: {checkpoint['avg_reward']:.3f}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def get_summary(self) -> dict:
        """Get training summary."""
        return {
            'experiment_name': self.experiment_name,
            'best_iteration': self.best_iteration,
            'best_success_rate': self.best_success_rate,
            'best_avg_reward': self.best_avg_reward,
            'environment': 'MiniGrid-FourRooms-v0'
        }

def collect_experience(env, agent, buffer, device, steps=2000):
    """Collect experience from FourRooms environment (longer episodes)."""
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
        
        # Reset if done (FourRooms episodes can be much longer)
        if done or episode_steps >= 200:  # Increased max episode length
            episode_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0
            episode_steps = 0
            obs = env.reset()
            h_t = agent.init_hidden(batch_size=1, device=device)
        else:
            obs = next_obs
    
    return len(buffer), episode_rewards

def train_world_model(agent, buffer, device, num_updates=100, batch_size=64):
    """Train world model (more updates for complex environment)."""
    if not buffer.is_ready(batch_size):
        return {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    
    world_model_optimizer = optim.Adam(agent.rssm.parameters(), lr=3e-4)
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    
    for _ in range(num_updates):
        # Sample sequences for world model training
        seq_len = 20  # Longer sequences for FourRooms
        batch = buffer.sample_sequences(batch_size // 4, seq_len)
        
        obs = batch['observations'].to(device)
        actions = batch['actions'].to(device)
        
        # Compute world model loss
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

def train_reward_model(agent, buffer, device, num_updates=150, batch_size=64):
    """Train reward model (more updates for sparse rewards)."""
    if not buffer.is_ready(batch_size):
        return 0.0
    
    reward_optimizer = optim.Adam(agent.reward_net.parameters(), lr=3e-4)
    total_loss = 0.0
    
    for _ in range(num_updates):
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

def train_continue_model(agent, buffer, device, num_updates=75, batch_size=64):
    """Train continue model."""
    if not buffer.is_ready(batch_size):
        return 0.0
    
    continue_optimizer = optim.Adam(agent.continue_net.parameters(), lr=3e-4)
    total_loss = 0.0
    
    for _ in range(num_updates):
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

def train_policy_value(agent, buffer, device, num_updates=75, imagination_horizon=20):
    """Train policy with longer imagination horizon for complex environment."""
    if not buffer.is_ready(32):
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    
    policy_optimizer = optim.Adam([
        {'params': agent.policy_net.parameters(), 'lr': 3e-4},
        {'params': agent.value_net.parameters(), 'lr': 3e-4}
    ])
    
    total_losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'mean_return': 0.0}
    
    for _ in range(num_updates):
        # Sample starting states
        batch = buffer.sample_batch(32)
        obs = batch['observations'].to(device)
        
        # Get starting latent states
        h_start = agent.init_hidden(batch_size=32, device=device)
        z_means, z_stds = agent.rssm.encode(obs, h_start)
        z_start = agent.rssm.sample_latent(z_means, z_stds)
        
        # Imagine trajectory with longer horizon for complex planning
        trajectory = agent.imagine_trajectory(z_start, h_start, steps=imagination_horizon)
        
        # Compute policy loss
        loss_dict = agent.compute_policy_loss(trajectory)
        
        # Total loss
        total_loss = (loss_dict['policy_loss'] + 
                     loss_dict['value_loss'] - 
                     0.02 * loss_dict['entropy'])  # Higher entropy for exploration
        
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

def evaluate_agent(env, agent, device, num_episodes=20):
    """Evaluate agent on FourRooms (more episodes for stability)."""
    episode_rewards = []
    success_count = 0
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        h_t = agent.init_hidden(batch_size=1, device=device)
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 200:  # Longer max episode length
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _, h_t = agent.act(obs_tensor, h_t, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, done, info = env.step(action_np)
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                if reward > 0.5:  # Success in FourRooms
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
    
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    avg_length = np.mean(episode_lengths)
    
    return avg_reward, success_rate, avg_length

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🏠 Weekend Challenge: DreamerV3 on MiniGrid-FourRooms-v0")
    print(f"Using device: {device}")
    
    # Initialize weekend checkpointer (longer patience for complex environment)
    checkpointer = WeekendCheckpointer(save_dir="weekend_fourrooms_experiments", patience=75)
    
    # Initialize wandb for weekend monitoring
    if WANDB_AVAILABLE:
        wandb.init(
            project="dreamer-v3-weekend-challenge",
            name="fourrooms-weekend-run",
            config={
                "environment": "MiniGrid-FourRooms-v0",
                "algorithm": "DreamerV3",
                "challenge": "weekend_fourrooms",
                "horizon": 20,
                "sequence_length": 20,
                "patience": 75,
                "max_episode_length": 200,
            }
        )
    
    # Environment (the challenge!)
    env = MiniGridEnvWrapper(env_name="MiniGrid-FourRooms-v0")
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()
    print(f"🏠 FourRooms obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    # DreamerV3 Agent (optimized for complex environment)
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        rssm_hidden_dim=256,    # Larger for complex environment
        rssm_latent_dim=128,    # Larger latent space
        policy_hidden_dim=512,  # Larger policy network
        value_hidden_dim=512,   # Larger value network
        reward_hidden_dim=256,
        continue_hidden_dim=256,
        num_reward_bins=41,
        num_value_bins=41
    ).to(device)
    
    # Larger replay buffer for complex environment
    buffer = DreamerReplayBuffer(capacity=100000, obs_dim=obs_dim, action_dim=action_dim)
    
    # Extended training parameters for weekend run
    num_iterations = 1000      # Much longer training
    collect_steps = 2000       # More steps per iteration
    world_model_updates = 100  # More world model updates
    reward_model_updates = 150 # More reward model updates (sparse rewards!)
    continue_model_updates = 75
    policy_updates = 75
    eval_frequency = 25        # Less frequent evaluation
    imagination_horizon = 20   # Longer horizon for complex planning
    
    print(f"🚀 Starting weekend FourRooms challenge...")
    print(f"📈 Extended training: {num_iterations} iterations")
    print(f"🧠 Imagination horizon: {imagination_horizon} steps")
    print(f"⏰ This will run over the weekend - check back Monday!")
    
    for iteration in range(num_iterations):
        # 1. Collect experience from complex environment
        buffer_size, episode_rewards = collect_experience(env, agent, buffer, device, collect_steps)
        
        if len(episode_rewards) > 0:
            recent_reward = np.mean(episode_rewards)
            if iteration % 5 == 0:  # Less frequent logging
                print(f"Iteration {iteration}: Recent episode reward: {recent_reward:.3f}")
        
        if not buffer.is_ready(2000):  # Larger minimum buffer
            continue
        
        # 2. Train world model with more updates
        wm_losses = train_world_model(agent, buffer, device, world_model_updates)
        
        # 3. Train reward model with more updates (sparse rewards!)
        rm_loss = train_reward_model(agent, buffer, device, reward_model_updates)
        
        # 4. Train continue model
        cm_loss = train_continue_model(agent, buffer, device, continue_model_updates)
        
        # 5. Train policy with longer imagination horizon
        pv_losses = train_policy_value(agent, buffer, device, policy_updates, imagination_horizon)
        
        # 6. Evaluate agent less frequently
        if iteration % eval_frequency == 0:
            avg_reward, success_rate, avg_length = evaluate_agent(env, agent, device)
            
            # Prepare metrics
            metrics = {
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
                'avg_episode_length': avg_length,
                'buffer_size': buffer_size
            }
            
            # Save checkpoint if improved
            improved = checkpointer.save_checkpoint(agent, iteration, success_rate, avg_reward, metrics)
            
            print(f"\n🏠 FourRooms Iteration {iteration}:")
            print(f"  Buffer: {buffer_size}, Avg Episode Length: {avg_length:.1f}")
            print(f"  World Model: {wm_losses['total_loss']:.4f}, Reward: {rm_loss:.4f}")
            print(f"  Policy: {pv_losses['policy_loss']:.4f}, Value: {pv_losses['value_loss']:.4f}")
            print(f"  🎯 Success: {success_rate:.1%}, Avg Reward: {avg_reward:.3f}")
            print(f"  Return Scale: {agent.return_normalizer.scale:.3f}")
            
            # Log to wandb
            if WANDB_AVAILABLE:
                wandb.log({
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'avg_episode_length': avg_length,
                    'improved': improved,
                    **metrics
                })
            
            # Check for early stopping
            if checkpointer.should_stop():
                print(f"\n🛑 WEEKEND TRAINING COMPLETED!")
                print(f"   Early stopping triggered after {iteration} iterations")
                print(f"   Loading best model from iteration {checkpointer.best_iteration}")
                break
    
    # Final evaluation with best model
    print(f"\n🏁 WEEKEND CHALLENGE COMPLETED!")
    
    # Load best model
    checkpointer.load_best_model(agent)
    final_reward, final_success, final_length = evaluate_agent(env, agent, device, num_episodes=50)
    
    # Get summary
    summary = checkpointer.get_summary()
    
    print(f"\n📊 WEEKEND FOURROOMS RESULTS:")
    print(f"   Best Success Rate: {summary['best_success_rate']:.1%} (Iteration {summary['best_iteration']})")
    print(f"   Best Avg Reward: {summary['best_avg_reward']:.3f}")
    print(f"\n🎯 FINAL EVALUATION (50 episodes):")
    print(f"   Success Rate: {final_success:.1%}")
    print(f"   Avg Reward: {final_reward:.3f}")
    print(f"   Avg Episode Length: {final_length:.1f} steps")
    
    # Save weekend summary
    summary_path = checkpointer.save_dir / "weekend_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"🏠 Weekend Challenge: MiniGrid-FourRooms-v0\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Environment: MiniGrid-FourRooms-v0 (Complex Navigation)\n")
        f.write(f"Algorithm: DreamerV3 with all upgrades\n")
        f.write(f"Imagination Horizon: {imagination_horizon} steps\n\n")
        f.write(f"Best Performance:\n")
        f.write(f"  Success Rate: {summary['best_success_rate']:.1%} (Iteration {summary['best_iteration']})\n")
        f.write(f"  Avg Reward: {summary['best_avg_reward']:.3f}\n\n")
        f.write(f"Final Evaluation (50 episodes):\n")
        f.write(f"  Success Rate: {final_success:.1%}\n")
        f.write(f"  Avg Reward: {final_reward:.3f}\n")
        f.write(f"  Avg Episode Length: {final_length:.1f} steps\n\n")
        f.write(f"Challenge Assessment:\n")
        if final_success >= 0.8:
            f.write(f"  🏆 EXCELLENT! DreamerV3 mastered complex navigation!\n")
        elif final_success >= 0.5:
            f.write(f"  ✅ GOOD! DreamerV3 learned complex environment!\n")
        elif final_success >= 0.2:
            f.write(f"  📈 PROGRESS! DreamerV3 showing learning on hard task!\n")
        else:
            f.write(f"  🔬 RESEARCH! More work needed for complex environments!\n")
    
    print(f"\n📁 Weekend results saved to: {checkpointer.save_dir}")
    print(f"📄 See weekend_summary.txt for complete results")
    print(f"🏠 Enjoy your weekend! Check back Monday for results! 🎉")

if __name__ == "__main__":
    train() 