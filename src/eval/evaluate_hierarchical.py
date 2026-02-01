"""Evaluation script for Hierarchical Reinforcement Learning."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.algorithms.hierarchical_agent import HierarchicalAgent
from src.utils.logging_utils import setup_logging
from src.utils.seeding import set_seed


def load_trained_agent(checkpoint_path: str, config: Dict) -> HierarchicalAgent:
    """Load a trained hierarchical agent.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Agent configuration
        
    Returns:
        Loaded hierarchical agent
    """
    logger = logging.getLogger(__name__)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create agent
    agent = HierarchicalAgent(
        state_dim=4,  # CartPole state dimension
        action_dim=2,  # CartPole action dimension
        num_options=config.num_options,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        device=device,
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using untrained agent")
    
    return agent


def evaluate_hierarchical_agent(
    agent: HierarchicalAgent,
    env_name: str = "CartPole-v1",
    num_episodes: int = 100,
    seed: int = 42,
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate hierarchical agent with detailed metrics.
    
    Args:
        agent: Hierarchical agent to evaluate
        env_name: Name of the environment
        num_episodes: Number of evaluation episodes
        seed: Random seed
        render: Whether to render episodes
        
    Returns:
        Evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)
    env = Monitor(env)
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    option_usage: Dict[int, int] = {}
    option_transitions: List[Dict] = []
    
    logger.info(f"Evaluating hierarchical agent for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        episode_length = 0
        current_option = None
        option_start_step = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Track option usage and transitions
            if agent.current_option:
                new_option = agent.current_option
                if new_option != current_option:
                    # Option transition occurred
                    if current_option is not None:
                        option_transitions.append({
                            "episode": episode,
                            "from_option": current_option.option_id,
                            "to_option": new_option.option_id,
                            "step": episode_length,
                            "duration": episode_length - option_start_step,
                        })
                    current_option = new_option
                    option_start_step = episode_length
                
                option_usage[current_option.option_id] = option_usage.get(current_option.option_id, 0) + 1
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    # Compute metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean([r >= 195 for r in episode_rewards]),  # CartPole threshold
        "option_usage": option_usage,
        "option_transitions": option_transitions,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    
    logger.info("Evaluation Results:")
    logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"  Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    logger.info(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
    logger.info(f"  Option Usage: {option_usage}")
    logger.info(f"  Option Transitions: {len(option_transitions)}")
    
    return metrics


def compare_with_baseline(
    hierarchical_metrics: Dict[str, float],
    env_name: str = "CartPole-v1",
    num_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Compare hierarchical agent with PPO baseline.
    
    Args:
        hierarchical_metrics: Metrics from hierarchical agent
        env_name: Name of the environment
        num_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Comparison metrics
    """
    logger = logging.getLogger(__name__)
    
    # Create environment for baseline
    env = gym.make(env_name)
    env = Monitor(env)
    
    # Load or create PPO baseline
    baseline_path = "checkpoints/ppo_baseline.zip"
    if os.path.exists(baseline_path):
        model = PPO.load(baseline_path)
        logger.info("Loaded trained PPO baseline")
    else:
        logger.info("Training PPO baseline for comparison...")
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        model.save(baseline_path)
        logger.info("PPO baseline trained and saved")
    
    # Evaluate PPO baseline
    logger.info("Evaluating PPO baseline...")
    ppo_rewards, ppo_lengths = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, return_episode_rewards=True
    )
    
    # Compute PPO metrics
    ppo_metrics = {
        "mean_reward": np.mean(ppo_rewards),
        "std_reward": np.std(ppo_rewards),
        "mean_length": np.mean(ppo_lengths),
        "std_length": np.std(ppo_lengths),
        "success_rate": np.mean([r >= 195 for r in ppo_rewards]),
    }
    
    logger.info("PPO Baseline Results:")
    logger.info(f"  Mean Reward: {ppo_metrics['mean_reward']:.2f} ± {ppo_metrics['std_reward']:.2f}")
    logger.info(f"  Mean Length: {ppo_metrics['mean_length']:.2f} ± {ppo_metrics['std_length']:.2f}")
    logger.info(f"  Success Rate: {ppo_metrics['success_rate']:.2%}")
    
    # Compute comparison
    comparison = {
        "hierarchical": hierarchical_metrics,
        "ppo": ppo_metrics,
        "reward_improvement": hierarchical_metrics["mean_reward"] - ppo_metrics["mean_reward"],
        "length_improvement": hierarchical_metrics["mean_length"] - ppo_metrics["mean_length"],
        "success_rate_improvement": hierarchical_metrics["success_rate"] - ppo_metrics["success_rate"],
    }
    
    logger.info("Comparison Results:")
    logger.info(f"  Reward Improvement: {comparison['reward_improvement']:.2f}")
    logger.info(f"  Length Improvement: {comparison['length_improvement']:.2f}")
    logger.info(f"  Success Rate Improvement: {comparison['success_rate_improvement']:.2%}")
    
    env.close()
    return comparison


def plot_evaluation_results(metrics: Dict[str, float], output_dir: str = "assets") -> None:
    """Plot evaluation results.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot reward distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(metrics["episode_rewards"], bins=20, alpha=0.7, edgecolor="black")
    plt.title("Episode Rewards Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.axvline(metrics["mean_reward"], color="red", linestyle="--", label=f"Mean: {metrics['mean_reward']:.2f}")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(metrics["episode_lengths"], bins=20, alpha=0.7, edgecolor="black")
    plt.title("Episode Lengths Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.axvline(metrics["mean_length"], color="red", linestyle="--", label=f"Mean: {metrics['mean_length']:.2f}")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    option_ids = list(metrics["option_usage"].keys())
    option_counts = list(metrics["option_usage"].values())
    plt.bar(option_ids, option_counts, alpha=0.7, edgecolor="black")
    plt.title("Option Usage")
    plt.xlabel("Option ID")
    plt.ylabel("Usage Count")
    
    plt.subplot(2, 2, 4)
    episodes = range(len(metrics["episode_rewards"]))
    plt.plot(episodes, metrics["episode_rewards"], alpha=0.7)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.axhline(195, color="red", linestyle="--", label="Success Threshold")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}/evaluation_results.png")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical RL Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with PPO baseline")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({
            "num_options": 2,
            "learning_rate": 3e-4,
            "gamma": 0.99,
        })
    
    # Set seed
    set_seed(args.seed)
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load trained agent
    agent = load_trained_agent(args.checkpoint, config)
    
    # Evaluate hierarchical agent
    hierarchical_metrics = evaluate_hierarchical_agent(
        agent=agent,
        env_name=args.env,
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
    )
    
    # Compare with baseline if requested
    if args.compare_baseline:
        comparison = compare_with_baseline(
            hierarchical_metrics=hierarchical_metrics,
            env_name=args.env,
            num_episodes=args.episodes,
            seed=args.seed,
        )
    
    # Generate plots if requested
    if args.plot:
        plot_evaluation_results(hierarchical_metrics)


if __name__ == "__main__":
    main()
