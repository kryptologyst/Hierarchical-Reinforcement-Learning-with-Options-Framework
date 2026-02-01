# Hierarchical Reinforcement Learning with Options Framework

A research-ready implementation of hierarchical reinforcement learning using the options framework. This project demonstrates how agents can learn to decompose complex tasks into reusable skills (options) and coordinate high-level decision making with low-level action execution.

## Overview

This implementation features:

- **Options Framework**: High-level policies select reusable skills, low-level policies execute primitive actions
- **Modern RL Stack**: Built on Gymnasium, PyTorch, and Stable-Baselines3
- **Comprehensive Evaluation**: Detailed metrics, baselines, and ablation studies
- **Interactive Demo**: Streamlit-based visualization and analysis tools
- **Production-Ready Structure**: Clean code, type hints, tests, and documentation

## Safety Notice

**This is a research/educational demonstration and is NOT intended for production control of real systems.** The algorithms and models shown here are for learning and experimentation purposes only.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Hierarchical-Reinforcement-Learning-with-Options-Framework.git
cd Hierarchical-Reinforcement-Learning-with-Options-Framework

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Training

```bash
# Train hierarchical agent
python src/train/train_hierarchical.py --episodes 1000 --env CartPole-v1

# Train with custom configuration
python src/train/train_hierarchical.py --config configs/train.yaml --episodes 2000

# Train with baseline comparison
python src/train/train_hierarchical.py --train-baseline --episodes 1000
```

### Evaluation

```bash
# Evaluate trained agent
python src/eval/evaluate_hierarchical.py --checkpoint checkpoints/final_model.pt --episodes 100

# Compare with baseline
python src/eval/evaluate_hierarchical.py --checkpoint checkpoints/final_model.pt --compare-baseline --plot
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_demo.py
```

## Project Structure

```
hierarchical-rl/
├── src/                          # Source code
│   ├── algorithms/              # RL algorithms
│   │   └── hierarchical_agent.py
│   ├── train/                   # Training scripts
│   │   └── train_hierarchical.py
│   ├── eval/                    # Evaluation scripts
│   │   └── evaluate_hierarchical.py
│   └── utils/                   # Utility functions
│       ├── seeding.py
│       └── logging_utils.py
├── configs/                     # Configuration files
│   └── train.yaml
├── demo/                        # Interactive demos
│   └── streamlit_demo.py
├── tests/                       # Test suite
│   └── test_hierarchical_agent.py
├── scripts/                     # Utility scripts
├── notebooks/                   # Jupyter notebooks
├── assets/                      # Generated plots and results
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project configuration
└── README.md                    # This file
```

## Algorithm Details

### Options Framework

The implementation uses the options framework where:

1. **Options** represent reusable skills with:
   - Initiation set: States where the option can be started
   - Policy: How to behave while the option is active
   - Termination condition: When the option should terminate

2. **High-Level Policy**: Selects which option to use based on current state
3. **Low-Level Policy**: Executes primitive actions within the selected option

### Current Options

- **Option 0 (Balance)**: Maintains pole balance
  - Can be initiated in any state
  - Terminates when pole angle or cart position exceeds thresholds
  
- **Option 1 (Position)**: Controls cart position
  - Can only be initiated when near center
  - Terminates when cart moves too far from center

### Training Algorithm

The agent uses policy gradient methods to learn both policies simultaneously:

1. Collect experience using current policies
2. Compute advantages using reward-to-go
3. Update both high-level and low-level policies using policy gradient
4. Repeat until convergence

## Environment Support

Currently supports:
- **CartPole-v1**: Classic control task for balance learning
- **MountainCar-v0**: Sparse reward navigation task
- **Acrobot-v1**: Underactuated swing-up task

## Evaluation Metrics

The evaluation includes:

- **Learning Metrics**: Average return, success rate, sample efficiency
- **Stability Metrics**: Reward variance, convergence analysis
- **Hierarchical Metrics**: Option usage patterns, transition analysis
- **Baseline Comparison**: Performance vs. standard RL algorithms

### Expected Performance (CartPole-v1)

| Metric | Hierarchical RL | PPO Baseline |
|--------|----------------|--------------|
| Mean Reward | 450+ | 400+ |
| Success Rate | 90%+ | 85%+ |
| Sample Efficiency | Moderate | Baseline |

## Configuration

Training can be customized via YAML configuration files:

```yaml
# configs/train.yaml
seed: 42
num_options: 2
learning_rate: 3e-4
gamma: 0.99
batch_size: 32
num_episodes: 1000
env_name: "CartPole-v1"
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_hierarchical_agent.py
```

### Code Quality

```bash
# Format code
black src/ tests/ demo/

# Lint code
ruff check src/ tests/ demo/

# Type checking (if mypy is installed)
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Advanced Features

### Custom Options

You can define custom options by modifying the `_create_options` method in `HierarchicalAgent`:

```python
def _create_options(self) -> List[Option]:
    options = []
    
    # Custom option: Aggressive control
    options.append(Option(
        option_id=2,
        name="aggressive",
        initiation_set=lambda state: abs(state[2]) > 0.1,  # Large angle
        termination_condition=lambda state: abs(state[2]) < 0.05,  # Small angle
    ))
    
    return options
```

### Environment Wrappers

Add custom environment wrappers for preprocessing:

```python
from gymnasium.wrappers import NormalizeObservation, FrameStack

env = gym.make("CartPole-v1")
env = NormalizeObservation(env)
env = FrameStack(env, 4)
```

### Logging and Monitoring

The project supports multiple logging backends:

- **TensorBoard**: `tensorboard --logdir tensorboard_logs`
- **Weights & Biases**: Set `WANDB_API_KEY` environment variable
- **MLflow**: Configure MLflow tracking server

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: Enable vectorized environments or use GPU
3. **Poor Performance**: Check hyperparameters, increase training episodes
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

1. Use GPU acceleration when available
2. Enable vectorized environments for faster data collection
3. Tune hyperparameters for your specific environment
4. Use curriculum learning for complex tasks

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hierarchical_rl,
  title={Hierarchical Reinforcement Learning with Options Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Hierarchical-Reinforcement-Learning-with-Options-Framework}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the RL environments
- PyTorch team for the deep learning framework
- Stable-Baselines3 for baseline implementations
- The RL research community for foundational work on hierarchical RL

## References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
2. Bacon, P. L., Harb, J., & Precup, D. (2017). The option-critic architecture.
3. Nachum, O., et al. (2018). Data-efficient hierarchical reinforcement learning.
4. Vezhnevets, A., et al. (2017). Feudal networks for hierarchical reinforcement learning.
# Hierarchical-Reinforcement-Learning-with-Options-Framework
