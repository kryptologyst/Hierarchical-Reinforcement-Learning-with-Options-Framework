"""Interactive demo for Hierarchical Reinforcement Learning."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from omegaconf import OmegaConf

from src.algorithms.hierarchical_agent import HierarchicalAgent, Option
from src.utils.seeding import set_seed


class HierarchicalRLDemo:
    """Interactive demo for Hierarchical RL."""
    
    def __init__(self):
        """Initialize the demo."""
        self.agent: Optional[HierarchicalAgent] = None
        self.env: Optional[gym.Env] = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration."""
        config_path = "configs/train.yaml"
        if os.path.exists(config_path):
            return OmegaConf.load(config_path)
        else:
            return {
                "num_options": 2,
                "learning_rate": 3e-4,
                "gamma": 0.99,
            }
    
    def load_agent(self, checkpoint_path: str) -> bool:
        """Load a trained agent.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.agent = HierarchicalAgent(
                state_dim=4,  # CartPole
                action_dim=2,  # CartPole
                num_options=self.config.num_options,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                device=device,
            )
            
            if os.path.exists(checkpoint_path):
                self.agent.load_checkpoint(checkpoint_path)
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading agent: {e}")
            return False
    
    def create_env(self, env_name: str = "CartPole-v1") -> bool:
        """Create environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.env = gym.make(env_name)
            return True
        except Exception as e:
            st.error(f"Error creating environment: {e}")
            return False
    
    def run_episode(self, max_steps: int = 500, render: bool = False) -> Dict:
        """Run a single episode.
        
        Args:
            max_steps: Maximum number of steps
            render: Whether to render the episode
            
        Returns:
            Episode data
        """
        if self.agent is None or self.env is None:
            return {}
        
        state, _ = self.env.reset()
        done = False
        episode_data = {
            "states": [state.copy()],
            "actions": [],
            "rewards": [],
            "options": [],
            "option_transitions": [],
            "total_reward": 0,
            "length": 0,
        }
        
        current_option = None
        option_start_step = 0
        
        for step in range(max_steps):
            action = self.agent.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            done = done or truncated
            
            # Track option usage
            if self.agent.current_option:
                new_option = self.agent.current_option
                if new_option != current_option:
                    if current_option is not None:
                        episode_data["option_transitions"].append({
                            "step": step,
                            "from_option": current_option.option_id,
                            "to_option": new_option.option_id,
                            "duration": step - option_start_step,
                        })
                    current_option = new_option
                    option_start_step = step
                
                episode_data["options"].append(current_option.option_id)
            else:
                episode_data["options"].append(-1)  # No option active
            
            episode_data["states"].append(next_state.copy())
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["total_reward"] += reward
            episode_data["length"] += 1
            
            state = next_state
            
            if done:
                break
        
        return episode_data
    
    def run_evaluation(self, num_episodes: int = 10) -> Dict:
        """Run evaluation over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        option_usage = {}
        all_option_transitions = []
        
        for episode in range(num_episodes):
            episode_data = self.run_episode()
            
            episode_rewards.append(episode_data["total_reward"])
            episode_lengths.append(episode_data["length"])
            
            # Count option usage
            for option_id in episode_data["options"]:
                if option_id >= 0:
                    option_usage[option_id] = option_usage.get(option_id, 0) + 1
            
            all_option_transitions.extend(episode_data["option_transitions"])
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "option_usage": option_usage,
            "option_transitions": all_option_transitions,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": np.mean([r >= 195 for r in episode_rewards]),
        }


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Hierarchical RL Demo",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 Hierarchical Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases a hierarchical reinforcement learning agent using the options framework.
    The agent learns to balance high-level decision making (selecting options/skills) with 
    low-level action execution within those options.
    """)
    
    # Initialize demo
    if "demo" not in st.session_state:
        st.session_state.demo = HierarchicalRLDemo()
    
    demo = st.session_state.demo
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Checkpoint selection
    checkpoint_dir = Path("checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    
    if checkpoint_files:
        checkpoint_options = ["None"] + [str(f) for f in checkpoint_files]
        selected_checkpoint = st.sidebar.selectbox(
            "Select Checkpoint",
            checkpoint_options,
            index=0,
        )
        
        if selected_checkpoint != "None":
            if st.sidebar.button("Load Agent"):
                with st.spinner("Loading agent..."):
                    success = demo.load_agent(selected_checkpoint)
                    if success:
                        st.sidebar.success("Agent loaded successfully!")
                    else:
                        st.sidebar.error("Failed to load agent")
    else:
        st.sidebar.warning("No checkpoints found. Train an agent first.")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        index=0,
    )
    
    if st.sidebar.button("Create Environment"):
        with st.spinner("Creating environment..."):
            success = demo.create_env(env_name)
            if success:
                st.sidebar.success("Environment created successfully!")
            else:
                st.sidebar.error("Failed to create environment")
    
    # Main content
    if demo.agent is None:
        st.warning("Please load an agent checkpoint first.")
        st.info("""
        To get started:
        1. Train an agent using the training script
        2. Select a checkpoint from the sidebar
        3. Click 'Load Agent' to load the trained model
        """)
        return
    
    if demo.env is None:
        st.warning("Please create an environment first.")
        return
    
    # Tabs for different demo features
    tab1, tab2, tab3, tab4 = st.tabs(["Single Episode", "Evaluation", "Option Analysis", "About"])
    
    with tab1:
        st.header("Single Episode Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            max_steps = st.slider("Max Steps", 100, 1000, 500)
            render_episode = st.checkbox("Render Episode (if supported)")
        
        with col2:
            if st.button("Run Episode", key="run_single"):
                with st.spinner("Running episode..."):
                    episode_data = demo.run_episode(max_steps=max_steps, render=render_episode)
                
                if episode_data:
                    st.success(f"Episode completed! Reward: {episode_data['total_reward']:.2f}, Length: {episode_data['length']}")
                    
                    # Plot episode trajectory
                    fig = go.Figure()
                    
                    # Plot state components
                    states = np.array(episode_data["states"])
                    for i, label in enumerate(["Position", "Velocity", "Angle", "Angular Velocity"]):
                        fig.add_trace(go.Scatter(
                            y=states[:, i],
                            mode='lines',
                            name=label,
                        ))
                    
                    fig.update_layout(
                        title="State Trajectory",
                        xaxis_title="Step",
                        yaxis_title="State Value",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot option usage
                    if episode_data["options"]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=episode_data["options"],
                            mode='lines+markers',
                            name="Active Option",
                        ))
                        fig.update_layout(
                            title="Option Usage Over Time",
                            xaxis_title="Step",
                            yaxis_title="Option ID",
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Multi-Episode Evaluation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_episodes = st.slider("Number of Episodes", 5, 50, 10)
        
        with col2:
            if st.button("Run Evaluation", key="run_eval"):
                with st.spinner(f"Running {num_episodes} episodes..."):
                    eval_results = demo.run_evaluation(num_episodes)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Reward", f"{eval_results['mean_reward']:.2f}", f"±{eval_results['std_reward']:.2f}")
                
                with col2:
                    st.metric("Mean Length", f"{eval_results['mean_length']:.2f}", f"±{eval_results['std_length']:.2f}")
                
                with col3:
                    st.metric("Success Rate", f"{eval_results['success_rate']:.1%}")
                
                # Plot reward distribution
                fig = px.histogram(
                    x=eval_results["episode_rewards"],
                    nbins=20,
                    title="Episode Rewards Distribution",
                    labels={"x": "Reward", "y": "Frequency"},
                )
                fig.add_vline(x=195, line_dash="dash", line_color="red", annotation_text="Success Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot option usage
                if eval_results["option_usage"]:
                    fig = px.bar(
                        x=list(eval_results["option_usage"].keys()),
                        y=list(eval_results["option_usage"].values()),
                        title="Option Usage Count",
                        labels={"x": "Option ID", "y": "Usage Count"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Option Analysis")
        
        st.markdown("""
        ### Understanding Options
        
        Options in hierarchical RL represent reusable skills or sub-policies:
        
        - **Option 0 (Balance)**: Focuses on maintaining balance
          - Can be initiated in any state
          - Terminates when the pole angle or cart position exceeds thresholds
        
        - **Option 1 (Position)**: Focuses on cart position control
          - Can only be initiated when near the center
          - Terminates when cart moves too far from center
        
        The high-level policy learns when to select each option based on the current state,
        while the low-level policy learns how to execute primitive actions within each option.
        """)
        
        if demo.agent and demo.agent.options:
            st.subheader("Current Options")
            for option in demo.agent.options:
                st.write(f"**{option.name}** (ID: {option.option_id})")
                st.write(f"- Active: {option.active}")
                st.write(f"- Step count: {option.step_count}")
    
    with tab4:
        st.header("About Hierarchical RL")
        
        st.markdown("""
        ### What is Hierarchical Reinforcement Learning?
        
        Hierarchical RL breaks down complex tasks into simpler sub-tasks or hierarchies.
        This approach allows agents to:
        
        1. **Learn reusable skills**: Options represent skills that can be applied across different situations
        2. **Improve sample efficiency**: By learning at multiple levels of abstraction
        3. **Handle long-horizon tasks**: High-level planning combined with low-level execution
        4. **Enable transfer learning**: Skills learned in one context can be reused in others
        
        ### The Options Framework
        
        An option consists of three components:
        - **Initiation set**: States where the option can be started
        - **Policy**: How to behave while the option is active
        - **Termination condition**: When the option should terminate
        
        ### This Implementation
        
        This demo uses a two-level hierarchy:
        - **High-level policy**: Selects which option to use based on current state
        - **Low-level policy**: Executes primitive actions within the selected option
        
        The agent learns both policies simultaneously using policy gradient methods.
        
        ### Safety Notice
        
        **This is a research/educational demonstration and is NOT intended for production control of real systems.**
        The algorithms and models shown here are for learning and experimentation purposes only.
        """)


if __name__ == "__main__":
    main()
