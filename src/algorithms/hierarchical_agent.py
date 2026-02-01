"""Hierarchical Reinforcement Learning with Options Framework.

This module implements a hierarchical RL system using the options framework,
where high-level policies select options (skills) and low-level policies execute
primitive actions within those options.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class Option:
    """Represents an option (skill) in hierarchical RL.
    
    An option consists of:
    - Initiation set: states where the option can be started
    - Policy: how to behave while the option is active
    - Termination condition: when the option should terminate
    """
    
    def __init__(
        self,
        option_id: int,
        name: str,
        initiation_set: Optional[callable] = None,
        termination_condition: Optional[callable] = None,
    ):
        """Initialize an option.
        
        Args:
            option_id: Unique identifier for the option
            name: Human-readable name for the option
            initiation_set: Function that determines if option can be initiated
            termination_condition: Function that determines if option should terminate
        """
        self.option_id = option_id
        self.name = name
        self.initiation_set = initiation_set or (lambda state: True)
        self.termination_condition = termination_condition or (lambda state: False)
        self.active = False
        self.step_count = 0
        
    def can_initiate(self, state: np.ndarray) -> bool:
        """Check if this option can be initiated in the given state."""
        return self.initiation_set(state)
    
    def should_terminate(self, state: np.ndarray) -> bool:
        """Check if this option should terminate in the given state."""
        return self.termination_condition(state) or self.step_count >= 10  # Max steps
    
    def activate(self) -> None:
        """Activate this option."""
        self.active = True
        self.step_count = 0
        
    def deactivate(self) -> None:
        """Deactivate this option."""
        self.active = False
        self.step_count = 0
        
    def step(self) -> None:
        """Increment step count for this option."""
        self.step_count += 1


class HighLevelPolicy(nn.Module):
    """High-level policy that selects options based on current state."""
    
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        """Initialize the high-level policy.
        
        Args:
            state_dim: Dimension of the state space
            num_options: Number of available options
            hidden_dim: Hidden layer dimension
            device: Device to run on
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_options = num_options
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the high-level policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            Option logits
        """
        return self.network(state)
    
    def select_option(self, state: np.ndarray, available_options: List[Option]) -> Optional[Option]:
        """Select an option based on current state.
        
        Args:
            state: Current state
            available_options: List of available options
            
        Returns:
            Selected option or None if no valid options
        """
        valid_options = [opt for opt in available_options if opt.can_initiate(state)]
        if not valid_options:
            return None
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.forward(state_tensor)
            # Mask out unavailable options
            mask = torch.zeros(self.num_options).to(self.device)
            for opt in valid_options:
                mask[opt.option_id] = 1.0
            masked_logits = logits + (mask - 1) * 1e9
            
            dist = Categorical(logits=masked_logits)
            option_id = dist.sample().item()
            
        return next((opt for opt in valid_options if opt.option_id == option_id), None)


class LowLevelPolicy(nn.Module):
    """Low-level policy that executes primitive actions within an option."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        option_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        """Initialize the low-level policy.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            option_dim: Dimension of option encoding
            hidden_dim: Hidden layer dimension
            device: Device to run on
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, state: torch.Tensor, option_encoding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the low-level policy.
        
        Args:
            state: Current state tensor
            option_encoding: One-hot encoding of current option
            
        Returns:
            Action logits
        """
        x = torch.cat([state, option_encoding], dim=-1)
        return self.network(x)
    
    def select_action(self, state: np.ndarray, option: Option) -> int:
        """Select a primitive action based on state and current option.
        
        Args:
            state: Current state
            option: Current active option
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        option_encoding = torch.zeros(1, self.option_dim).to(self.device)
        option_encoding[0, option.option_id] = 1.0
        
        with torch.no_grad():
            logits = self.forward(state_tensor, option_encoding)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            
        return action


class HierarchicalAgent:
    """Hierarchical RL agent using options framework."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_options: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        """Initialize the hierarchical agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            num_options: Number of available options
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.device = device
        
        # Initialize policies
        self.high_level_policy = HighLevelPolicy(state_dim, num_options, device=device)
        self.low_level_policy = LowLevelPolicy(state_dim, action_dim, num_options, device=device)
        
        # Initialize optimizers
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(), lr=learning_rate
        )
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(), lr=learning_rate
        )
        
        # Initialize options
        self.options = self._create_options()
        self.current_option: Optional[Option] = None
        
        # Experience buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        
    def _create_options(self) -> List[Option]:
        """Create the set of available options."""
        options = []
        
        # Option 0: Balance-focused policy
        options.append(Option(
            option_id=0,
            name="balance",
            initiation_set=lambda state: True,  # Can be initiated anywhere
            termination_condition=lambda state: abs(state[0]) > 2.4 or abs(state[2]) > 0.2,
        ))
        
        # Option 1: Position-focused policy
        options.append(Option(
            option_id=1,
            name="position",
            initiation_set=lambda state: abs(state[0]) < 1.0,  # Only when near center
            termination_condition=lambda state: abs(state[0]) > 1.5,
        ))
        
        return options
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using hierarchical decision making.
        
        Args:
            state: Current state
            
        Returns:
            Selected primitive action
        """
        # Check if we need to select a new option
        if (self.current_option is None or 
            not self.current_option.active or 
            self.current_option.should_terminate(state)):
            
            # Select new option
            self.current_option = self.high_level_policy.select_option(state, self.options)
            if self.current_option is not None:
                self.current_option.activate()
        
        # Select primitive action within current option
        if self.current_option is not None:
            action = self.low_level_policy.select_action(state, self.current_option)
            self.current_option.step()
        else:
            # Fallback to random action if no valid option
            action = np.random.randint(0, self.action_dim)
            
        return action
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        option_id: Optional[int] = None,
    ) -> None:
        """Store experience in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            option_id: ID of the option that was active
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "option_id": option_id,
        }
        self.experience_buffer.append(experience)
    
    def update_policies(self, batch_size: int = 32) -> Dict[str, float]:
        """Update both high-level and low-level policies.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Dictionary of losses
        """
        if len(self.experience_buffer) < batch_size:
            return {"high_level_loss": 0.0, "low_level_loss": 0.0}
        
        # Sample batch
        batch = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch_experiences = [self.experience_buffer[i] for i in batch]
        
        # Prepare data
        states = torch.FloatTensor([exp["state"] for exp in batch_experiences]).to(self.device)
        actions = torch.LongTensor([exp["action"] for exp in batch_experiences]).to(self.device)
        rewards = torch.FloatTensor([exp["reward"] for exp in batch_experiences]).to(self.device)
        next_states = torch.FloatTensor([exp["next_state"] for exp in batch_experiences]).to(self.device)
        dones = torch.BoolTensor([exp["done"] for exp in batch_experiences]).to(self.device)
        option_ids = torch.LongTensor([exp["option_id"] for exp in batch_experiences]).to(self.device)
        
        # Update low-level policy (simplified policy gradient)
        option_encodings = torch.zeros(batch_size, self.num_options).to(self.device)
        option_encodings[range(batch_size), option_ids] = 1.0
        
        action_logits = self.low_level_policy(states, option_encodings)
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Compute advantages (simplified)
        advantages = rewards - rewards.mean()
        
        # Policy gradient loss
        selected_log_probs = action_log_probs[range(batch_size), actions]
        low_level_loss = -(selected_log_probs * advantages).mean()
        
        # Update low-level policy
        self.low_level_optimizer.zero_grad()
        low_level_loss.backward()
        self.low_level_optimizer.step()
        
        # Update high-level policy (simplified)
        option_logits = self.high_level_policy(states)
        option_probs = F.softmax(option_logits, dim=-1)
        option_log_probs = F.log_softmax(option_logits, dim=-1)
        
        # Select option log probabilities
        selected_option_log_probs = option_log_probs[range(batch_size), option_ids]
        high_level_loss = -(selected_option_log_probs * advantages).mean()
        
        # Update high-level policy
        self.high_level_optimizer.zero_grad()
        high_level_loss.backward()
        self.high_level_optimizer.step()
        
        return {
            "high_level_loss": high_level_loss.item(),
            "low_level_loss": low_level_loss.item(),
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "high_level_policy": self.high_level_policy.state_dict(),
            "low_level_policy": self.low_level_policy.state_dict(),
            "high_level_optimizer": self.high_level_optimizer.state_dict(),
            "low_level_optimizer": self.low_level_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.high_level_policy.load_state_dict(checkpoint["high_level_policy"])
        self.low_level_policy.load_state_dict(checkpoint["low_level_policy"])
        self.high_level_optimizer.load_state_dict(checkpoint["high_level_optimizer"])
        self.low_level_optimizer.load_state_dict(checkpoint["low_level_optimizer"])
        logger.info(f"Checkpoint loaded from {filepath}")
