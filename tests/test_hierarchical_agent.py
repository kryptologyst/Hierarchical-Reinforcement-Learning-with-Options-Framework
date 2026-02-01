"""Tests for hierarchical RL components."""

import pytest
import torch
import numpy as np

from src.algorithms.hierarchical_agent import HierarchicalAgent, Option, HighLevelPolicy, LowLevelPolicy


class TestOption:
    """Test cases for Option class."""
    
    def test_option_creation(self):
        """Test option creation."""
        option = Option(option_id=0, name="test_option")
        assert option.option_id == 0
        assert option.name == "test_option"
        assert not option.active
        assert option.step_count == 0
    
    def test_option_activation(self):
        """Test option activation and deactivation."""
        option = Option(option_id=0, name="test_option")
        
        option.activate()
        assert option.active
        assert option.step_count == 0
        
        option.step()
        assert option.step_count == 1
        
        option.deactivate()
        assert not option.active
        assert option.step_count == 0
    
    def test_option_termination(self):
        """Test option termination conditions."""
        option = Option(
            option_id=0,
            name="test_option",
            termination_condition=lambda state: state[0] > 1.0
        )
        
        # Should not terminate
        assert not option.should_terminate(np.array([0.5, 0.0]))
        
        # Should terminate due to custom condition
        assert option.should_terminate(np.array([1.5, 0.0]))
        
        # Should terminate due to max steps
        for _ in range(10):
            option.step()
        assert option.should_terminate(np.array([0.5, 0.0]))


class TestHighLevelPolicy:
    """Test cases for HighLevelPolicy class."""
    
    def test_policy_creation(self):
        """Test policy creation."""
        policy = HighLevelPolicy(state_dim=4, num_options=2)
        assert policy.state_dim == 4
        assert policy.num_options == 2
    
    def test_policy_forward(self):
        """Test forward pass."""
        policy = HighLevelPolicy(state_dim=4, num_options=2)
        state = torch.randn(1, 4)
        output = policy(state)
        assert output.shape == (1, 2)
    
    def test_option_selection(self):
        """Test option selection."""
        policy = HighLevelPolicy(state_dim=4, num_options=2)
        
        # Create test options
        options = [
            Option(option_id=0, name="option0"),
            Option(option_id=1, name="option1"),
        ]
        
        state = np.array([0.0, 0.0, 0.0, 0.0])
        selected_option = policy.select_option(state, options)
        
        assert selected_option is not None
        assert selected_option in options


class TestLowLevelPolicy:
    """Test cases for LowLevelPolicy class."""
    
    def test_policy_creation(self):
        """Test policy creation."""
        policy = LowLevelPolicy(state_dim=4, action_dim=2, option_dim=2)
        assert policy.state_dim == 4
        assert policy.action_dim == 2
        assert policy.option_dim == 2
    
    def test_policy_forward(self):
        """Test forward pass."""
        policy = LowLevelPolicy(state_dim=4, action_dim=2, option_dim=2)
        state = torch.randn(1, 4)
        option_encoding = torch.randn(1, 2)
        output = policy(state, option_encoding)
        assert output.shape == (1, 2)
    
    def test_action_selection(self):
        """Test action selection."""
        policy = LowLevelPolicy(state_dim=4, action_dim=2, option_dim=2)
        option = Option(option_id=0, name="test_option")
        
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = policy.select_action(state, option)
        
        assert action in [0, 1]


class TestHierarchicalAgent:
    """Test cases for HierarchicalAgent class."""
    
    def test_agent_creation(self):
        """Test agent creation."""
        agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.num_options == 2
        assert len(agent.options) == 2
    
    def test_action_selection(self):
        """Test action selection."""
        agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = agent.select_action(state)
        assert action in [0, 1]
    
    def test_experience_storage(self):
        """Test experience storage."""
        agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        
        initial_buffer_size = len(agent.experience_buffer)
        
        agent.store_experience(
            state=np.array([0.0, 0.0, 0.0, 0.0]),
            action=0,
            reward=1.0,
            next_state=np.array([0.1, 0.0, 0.0, 0.0]),
            done=False,
            option_id=0,
        )
        
        assert len(agent.experience_buffer) == initial_buffer_size + 1
    
    def test_policy_update(self):
        """Test policy update."""
        agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        
        # Add some experiences
        for _ in range(50):
            agent.store_experience(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=np.random.choice([True, False]),
                option_id=np.random.randint(0, 2),
            )
        
        # Update policies
        losses = agent.update_policies(batch_size=32)
        
        assert "high_level_loss" in losses
        assert "low_level_loss" in losses
        assert isinstance(losses["high_level_loss"], float)
        assert isinstance(losses["low_level_loss"], float)
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        agent.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Create new agent and load checkpoint
        new_agent = HierarchicalAgent(state_dim=4, action_dim=2, num_options=2)
        new_agent.load_checkpoint(str(checkpoint_path))
        
        # Verify models are loaded
        assert new_agent.high_level_policy is not None
        assert new_agent.low_level_policy is not None


if __name__ == "__main__":
    pytest.main([__file__])
