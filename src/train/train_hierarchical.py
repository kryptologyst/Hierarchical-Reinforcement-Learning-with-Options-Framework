"""Training script for Hierarchical Reinforcement Learning."""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from src.algorithms.hierarchical_agent import HierarchicalAgent
from src.utils.logging_utils import setup_logging
from src.utils.seeding import set_seed


def create_env(env_name: str, seed: int = 0) -> gym.Env:
    """Create and configure environment.
    
    Args:
        env_name: Name of the environment
        seed: Random seed
        
    Returns:
        Configured environment
    """
    env = gym.make(env_name)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_hierarchical_agent(
    config: Dict,
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    eval_freq: int = 100,
    save_freq: int = 200,
    device: str = "cpu",
) -> HierarchicalAgent:
    """Train hierarchical RL agent.
    
    Args:
        config: Training configuration
        env_name: Name of the environment
        num_episodes: Number of training episodes
        eval_freq: Frequency of evaluation
        save_freq: Frequency of saving checkpoints
        device: Device to run on
        
    Returns:
        Trained hierarchical agent
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Create environment
    env = create_env(env_name, config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize hierarchical agent
    agent = HierarchicalAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_options=config.num_options,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        device=device,
    )
    
    # Training metrics
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[Dict[str, float]] = []
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {num_episodes} episodes")
    logger.info(f"Environment: {env_name}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"Number of options: {config.num_options}")
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Select action using hierarchical agent
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            option_id = agent.current_option.option_id if agent.current_option else None
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                option_id=option_id,
            )
            
            # Update policies
            if len(agent.experience_buffer) >= config.batch_size:
                loss_dict = agent.update_policies(batch_size=config.batch_size)
                losses.append(loss_dict)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging and evaluation
        if episode % eval_freq == 0:
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            avg_length = np.mean(episode_lengths[-eval_freq:])
            
            logger.info(
                f"Episode {episode}: "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Avg Length: {avg_length:.2f}"
            )
            
            # Save checkpoint
            if episode % save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint_episode_{episode}.pt"
                agent.save_checkpoint(str(checkpoint_path))
                logger.info(f"Checkpoint saved at episode {episode}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    agent.save_checkpoint(str(final_path))
    
    # Save training metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
    }
    np.save(output_dir / "training_metrics.npy", metrics)
    
    logger.info("Training completed!")
    return agent


def evaluate_agent(
    agent: HierarchicalAgent,
    env_name: str = "CartPole-v1",
    num_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate hierarchical agent.
    
    Args:
        agent: Trained hierarchical agent
        env_name: Name of the environment
        num_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Create evaluation environment
    env = create_env(env_name, seed)
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    option_usage: Dict[int, int] = {}
    
    logger.info(f"Evaluating agent for {num_episodes} episodes")
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Track option usage
            if agent.current_option:
                option_id = agent.current_option.option_id
                option_usage[option_id] = option_usage.get(option_id, 0) + 1
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Compute metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean([r >= 195 for r in episode_rewards]),  # CartPole threshold
        "option_usage": option_usage,
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
    logger.info(f"  Option Usage: {option_usage}")
    
    return metrics


def train_baseline_ppo(
    env_name: str = "CartPole-v1",
    total_timesteps: int = 100000,
    device: str = "cpu",
) -> PPO:
    """Train PPO baseline for comparison.
    
    Args:
        env_name: Name of the environment
        total_timesteps: Total training timesteps
        device: Device to run on
        
    Returns:
        Trained PPO agent
    """
    logger = logging.getLogger(__name__)
    
    # Create vectorized environment
    env = make_vec_env(env_name, n_envs=4, seed=42)
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    logger.info("Training PPO baseline...")
    model.learn(total_timesteps=total_timesteps)
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Hierarchical RL Agent")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--train-baseline", action="store_true", help="Train PPO baseline")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        # Default configuration
        config = OmegaConf.create({
            "seed": args.seed,
            "num_options": 2,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "output_dir": "checkpoints",
        })
    
    # Set seed
    set_seed(config.seed)
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Train hierarchical agent
    agent = train_hierarchical_agent(
        config=config,
        env_name=args.env,
        num_episodes=args.episodes,
        device=device,
    )
    
    # Evaluate hierarchical agent
    eval_metrics = evaluate_agent(
        agent=agent,
        env_name=args.env,
        num_episodes=args.eval_episodes,
        seed=config.seed,
    )
    
    # Train and evaluate baseline if requested
    if args.train_baseline:
        logger.info("Training PPO baseline...")
        ppo_model = train_baseline_ppo(env_name=args.env, device=device)
        
        logger.info("Evaluating PPO baseline...")
        ppo_rewards, ppo_lengths = evaluate_policy(
            ppo_model, 
            create_env(args.env, config.seed), 
            n_eval_episodes=args.eval_episodes,
            return_episode_rewards=True
        )
        
        logger.info(f"PPO Baseline Results:")
        logger.info(f"  Mean Reward: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
        logger.info(f"  Mean Length: {np.mean(ppo_lengths):.2f} ± {np.std(ppo_lengths):.2f}")
        logger.info(f"  Success Rate: {np.mean([r >= 195 for r in ppo_rewards]):.2%}")


if __name__ == "__main__":
    main()
