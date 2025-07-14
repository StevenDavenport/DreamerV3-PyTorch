#!/usr/bin/env python3
"""
DreamerV3 Training with Smart Checkpointing and Early Stopping

Features:
- Save best model based on success rate
- Early stopping when performance degrades
- Automatic rollback to best checkpoint
- Performance tracking and monitoring
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

class ModelCheckpointer:
    """Handles model checkpointing with early stopping logic."""
    
    def __init__(self, save_dir: str = "checkpoints", patience: int = 20, experiment_name: str = None):
        # Create timestamped experiment folder
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dreamerv3_experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        
        # Performance tracking
        self.best_success_rate = -1.0
        self.best_avg_reward = -float('inf')
        self.iterations_without_improvement = 0
        self.best_iteration = -1
        
        # Performance history
        self.history = {
            'iterations': [],
            'success_rates': [],
            'avg_rewards': [],
            'world_model_losses': [],
            'reward_model_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'mean_imagined_returns': [],
            'return_scales': []
        }
        
        # Hyperparameters (will be set during training)
        self.hyperparams = {}
        
        print(f"📁 Experiment: {experiment_name}")
        print(f"📁 Checkpointer initialized: {self.save_dir}")
        print(f"🛡️ Early stopping patience: {self.patience} iterations")
    
    def should_save(self, success_rate: float, avg_reward: float) -> bool:
        """Determine if current model is better than previous best."""
        # Primary criterion: success rate
        if success_rate > self.best_success_rate:
            return True
        # Secondary criterion: if success rates equal, use avg reward
        elif success_rate == self.best_success_rate and avg_reward > self.best_avg_reward:
            return True
        return False
    
    def save_checkpoint(self, agent: DreamerV3Agent, iteration: int, 
                       success_rate: float, avg_reward: float, 
                       metrics: dict) -> bool:
        """Save model checkpoint if performance improved."""
        
        # Update history
        self.history['iterations'].append(iteration)
        self.history['success_rates'].append(success_rate)
        self.history['avg_rewards'].append(avg_reward)
        self.history['world_model_losses'].append(metrics.get('world_model_loss', 0))
        self.history['reward_model_losses'].append(metrics.get('reward_model_loss', 0))
        self.history['policy_losses'].append(metrics.get('policy_loss', 0))
        self.history['value_losses'].append(metrics.get('value_loss', 0))
        self.history['mean_imagined_returns'].append(metrics.get('mean_imagined_return', 0))
        self.history['return_scales'].append(metrics.get('return_scale', 1.0))
        
        improved = self.should_save(success_rate, avg_reward)
        
        if improved:
            # Save new best model
            self.best_success_rate = success_rate
            self.best_avg_reward = avg_reward
            self.best_iteration = iteration
            self.iterations_without_improvement = 0
            
            # Create descriptive model name
            model_name = f"dreamerv3_iter{iteration:03d}_success{success_rate:.0%}_reward{avg_reward:.3f}"
            
            checkpoint = {
                'iteration': iteration,
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'agent_state_dict': agent.state_dict(),
                'return_normalizer_scale': agent.return_normalizer.scale,
                'return_normalizer_initialized': agent.return_normalizer.initialized,
                'metrics': metrics,
                'hyperparams': self.hyperparams,
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name
            }
            
            # Save with descriptive name
            checkpoint_path = self.save_dir / f"{model_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Also save as "best_model.pt" for easy loading
            best_model_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            
            # Save simple text log
            self._save_simple_log(iteration, success_rate, avg_reward, metrics)
            
            print(f"💾 NEW BEST MODEL SAVED!")
            print(f"   Iteration: {iteration}")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Reward: {avg_reward:.3f}")
            print(f"   Model: {model_name}")
            print(f"   Saved to: {checkpoint_path}")
            
            return True
        else:
            self.iterations_without_improvement += 1
            print(f"⏳ No improvement for {self.iterations_without_improvement}/{self.patience} iterations")
            return False
    
    def should_stop(self) -> bool:
        """Check if training should stop due to lack of improvement."""
        return self.iterations_without_improvement >= self.patience
    
    def load_best_model(self, agent: DreamerV3Agent) -> bool:
        """Load the best saved model."""
        checkpoint_path = self.save_dir / "best_model.pt"
        
        if not checkpoint_path.exists():
            print(f"❌ No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            
            # Restore return normalizer state
            agent.return_normalizer.scale = checkpoint['return_normalizer_scale']
            agent.return_normalizer.initialized = checkpoint['return_normalizer_initialized']
            
            print(f"✅ Loaded best model from iteration {checkpoint['iteration']}")
            print(f"   Success Rate: {checkpoint['success_rate']:.1%}")
            print(f"   Avg Reward: {checkpoint['avg_reward']:.3f}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def set_hyperparams(self, hyperparams: dict):
        """Set hyperparameters for logging."""
        self.hyperparams = hyperparams
    
    def _save_simple_log(self, iteration: int, success_rate: float, avg_reward: float, metrics: dict):
        """Save simple readable training log."""
        log_path = self.save_dir / "training_log.txt"
        with open(log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"NEW BEST MODEL - Iteration {iteration}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Performance:\n")
            f.write(f"  Success Rate: {success_rate:.1%}\n")
            f.write(f"  Avg Reward: {avg_reward:.3f}\n")
            f.write(f"\nKey Metrics:\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write(f"\n")
    
    def get_summary(self) -> dict:
        """Get training summary."""
        return {
            'experiment_name': self.experiment_name,
            'best_iteration': self.best_iteration,
            'best_success_rate': self.best_success_rate,
            'best_avg_reward': self.best_avg_reward,
            'total_iterations': len(self.history['iterations']),
            'final_success_rate': self.history['success_rates'][-1] if self.history['success_rates'] else 0.0,
            'hyperparams': self.hyperparams
        }

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
        
        # Get latent states (single timestep, not sequences)
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
    """Train continue model."""
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
        
        # Imagine trajectory using DreamerV3 agent (shorter horizon for stability)
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
    
    # Initialize checkpointer with descriptive experiment name
    experiment_name = f"dreamerv3_minigrid_empty6x6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpointer = ModelCheckpointer(save_dir="dreamer_v3_experiments", patience=25, experiment_name=experiment_name)
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="dreamer-v3-minigrid",
            name="dreamerv3-with-checkpointing",
            config={
                "environment": "MiniGrid-Empty-6x6-v0",
                "algorithm": "DreamerV3",
                "device": device,
                "upgrades": ["symlog_two_hot", "return_normalization", "free_bits", "kl_balancing"],
                "checkpointing": True,
                "early_stopping_patience": 25,
                "imagination_horizon": 15,  # Full horizon as per DreamerV3 paper
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
    num_iterations = 300  # Max iterations
    collect_steps = 1000
    world_model_updates = 50
    reward_model_updates = 100
    continue_model_updates = 50
    policy_updates = 50
    eval_frequency = 20
    imagination_horizon = 15
    
    # Collect all hyperparameters
    hyperparams = {
        # Environment
        'environment': 'MiniGrid-Empty-6x6-v0',
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        
        # Agent Architecture
        'rssm_hidden_dim': 128,
        'rssm_latent_dim': 64,
        'policy_hidden_dim': 256,
        'value_hidden_dim': 256,
        'reward_hidden_dim': 256,
        'continue_hidden_dim': 256,
        'num_reward_bins': 41,
        'num_value_bins': 41,
        
        # Training
        'max_iterations': num_iterations,
        'collect_steps': collect_steps,
        'world_model_updates': world_model_updates,
        'reward_model_updates': reward_model_updates,
        'continue_model_updates': continue_model_updates,
        'policy_updates': policy_updates,
        'eval_frequency': eval_frequency,
        'imagination_horizon': imagination_horizon,
        
        # DreamerV3 Specific
        'symlog_two_hot': True,
        'return_normalization': True,
        'free_bits': 1.0,
        'kl_balancing_alpha': 0.8,
        
        # Optimization
        'world_model_lr': 3e-4,
        'reward_model_lr': 3e-4,
        'continue_model_lr': 3e-4,
        'policy_lr': 3e-4,
        'value_lr': 3e-4,
        'gradient_clip': 1.0,
        'entropy_coef': 0.01,
        
        # Buffer
        'buffer_capacity': 50000,
        'sequence_length': 16,
        
        # Checkpointing
        'early_stopping_patience': 25,
        'eval_episodes': 10,
        
        # Device and misc
        'device': device,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat()
    }
    
    # Set hyperparameters in checkpointer
    checkpointer.set_hyperparams(hyperparams)
    
    print("🚀 Starting DreamerV3 training with smart checkpointing...")
    print("💡 Will automatically save best models and stop when performance stops improving")
    
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
        
        # 3. Train reward model with symlog + two-hot encoding
        rm_loss = train_reward_model(agent, buffer, device, reward_model_updates)
        
        # 4. Train continue model
        cm_loss = train_continue_model(agent, buffer, device, continue_model_updates)
        
        # 5. Train policy and value using imagination with return normalization
        pv_losses = train_policy_value(agent, buffer, device, policy_updates, imagination_horizon)
        
        # 6. Evaluate agent and checkpoint
        if iteration % eval_frequency == 0:
            avg_reward, success_rate = evaluate_agent(env, agent, device)
            
            # Prepare metrics for checkpointer
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
                'buffer_size': buffer_size
            }
            
            # Save checkpoint if improved
            improved = checkpointer.save_checkpoint(agent, iteration, success_rate, avg_reward, metrics)
            
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
                log_dict = {
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'improved': improved,
                    'iterations_without_improvement': checkpointer.iterations_without_improvement,
                    **metrics
                }
                wandb.log(log_dict)
            
            # Check for early stopping
            if checkpointer.should_stop():
                print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                print(f"   No improvement for {checkpointer.patience} iterations")
                print(f"   Loading best model from iteration {checkpointer.best_iteration}")
                
                # Load best model
                if checkpointer.load_best_model(agent):
                    print("✅ Best model loaded successfully")
                break
    
    # Final evaluation with best model
    print(f"\n🏁 TRAINING COMPLETED!")
    
    # Load best model for final evaluation
    checkpointer.load_best_model(agent)
    final_reward, final_success = evaluate_agent(env, agent, device, num_episodes=50)
    
    # Get training summary
    summary = checkpointer.get_summary()
    
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Best iteration: {summary['best_iteration']}")
    print(f"   Best success rate: {summary['best_success_rate']:.1%}")
    print(f"   Best avg reward: {summary['best_avg_reward']:.3f}")
    print(f"   Total iterations: {summary['total_iterations']}")
    
    print(f"\n🎯 FINAL EVALUATION (50 episodes):")
    print(f"   Average Reward: {final_reward:.3f}")
    print(f"   Success Rate: {final_success:.1%}")
    
    if final_success > 0.8:
        print(f"\n🎉 EXCELLENT! DreamerV3 with checkpointing achieved high performance!")
    elif final_success > 0.5:
        print(f"\n✅ GOOD! DreamerV3 solved the task successfully!")
    else:
        print(f"\n⚠️ Partial success - checkpointing prevented complete collapse!")
    
    # Save simple final summary
    summary_path = checkpointer.save_dir / "final_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"DreamerV3 Experiment: {experiment_name}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Best Performance:\n")
        f.write(f"  Success Rate: {summary['best_success_rate']:.1%} (Iteration {summary['best_iteration']})\n")
        f.write(f"  Avg Reward: {summary['best_avg_reward']:.3f}\n\n")
        f.write(f"Final Evaluation (50 episodes):\n")
        f.write(f"  Success Rate: {final_success:.1%}\n")
        f.write(f"  Avg Reward: {final_reward:.3f}\n\n")
        f.write(f"Training completed after {summary['total_iterations']} iterations\n")
    
    print(f"📁 All results saved to: {checkpointer.save_dir}")
    print(f"📄 See final_summary.txt and training_log.txt for details")

if __name__ == "__main__":
    train() 