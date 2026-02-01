#!/usr/bin/env python3
"""Quick start script for training and evaluating hierarchical RL agent."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.utils.logging_utils import setup_logging


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        cmd: Command to run
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print("Error:", e.stderr)
        return False


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="Quick start for Hierarchical RL")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate the agent")
    parser.add_argument("--demo", action="store_true", help="Launch interactive demo")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", help="Checkpoint path")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    
    success_count = 0
    total_steps = 0
    
    if args.all or args.train:
        total_steps += 1
        cmd = f"python src/train/train_hierarchical.py --episodes {args.episodes} --train-baseline"
        if run_command(cmd, "Training Hierarchical RL Agent"):
            success_count += 1
    
    if args.all or args.eval:
        total_steps += 1
        if os.path.exists(args.checkpoint):
            cmd = f"python src/eval/evaluate_hierarchical.py --checkpoint {args.checkpoint} --compare-baseline --plot"
            if run_command(cmd, "Evaluating Trained Agent"):
                success_count += 1
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Please train the agent first or specify a valid checkpoint path.")
    
    if args.all or args.test:
        total_steps += 1
        cmd = "pytest tests/ -v"
        if run_command(cmd, "Running Test Suite"):
            success_count += 1
    
    if args.all or args.demo:
        total_steps += 1
        print(f"\n{'='*60}")
        print("Launching Interactive Demo")
        print(f"{'='*60}")
        print("The demo will open in your browser.")
        print("Make sure you have trained an agent first!")
        print("Press Ctrl+C to stop the demo.")
        
        try:
            cmd = "streamlit run demo/streamlit_demo.py"
            subprocess.run(cmd, shell=True)
            success_count += 1
        except KeyboardInterrupt:
            print("\nDemo stopped by user.")
        except Exception as e:
            print(f"Error launching demo: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("QUICK START SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("All steps completed successfully!")
        print("\nNext steps:")
        print("1. Check the 'assets/' directory for evaluation plots")
        print("2. Check the 'checkpoints/' directory for saved models")
        print("3. Run 'streamlit run demo/streamlit_demo.py' for interactive analysis")
    else:
        print("Some steps failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
