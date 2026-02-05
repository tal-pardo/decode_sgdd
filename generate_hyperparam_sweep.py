#!/usr/bin/env python3
"""
Generate a grid of hyperparameter configs for SLURM job array submission.

This script creates multiple YAML config files from a hyperparameter sweep grid.
Each config is named config_0.yaml, config_1.yaml, etc., for use with job arrays.

Usage:
    python generate_hyperparam_sweep.py --num-configs 12 --output-dir configs/hyperparam_sweep
"""

import argparse
import yaml
from pathlib import Path
from itertools import product
import shutil


def generate_sweep_configs(base_config_path, param_grid, output_dir, num_configs=None):
    """
    Generate hyperparameter sweep configs from a parameter grid.
    
    Args:
        base_config_path: Path to base config YAML file
        param_grid: Dict of parameter names -> list of values to try
        output_dir: Directory to save generated configs
        num_configs: If specified, only generate first N configs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    if num_configs is not None:
        combinations = combinations[:num_configs]
    
    print(f"Generating {len(combinations)} config files...")
    
    for idx, values in enumerate(combinations):
        config = base_config.copy()
        
        # Apply parameters to config
        for param_name, value in zip(param_names, values):
            if '.' in param_name:  # Nested parameter like "training.lr"
                parts = param_name.split('.')
                subconfig = config
                for part in parts[:-1]:
                    if part not in subconfig:
                        subconfig[part] = {}
                    subconfig = subconfig[part]
                subconfig[parts[-1]] = value
            else:
                config[param_name] = value
        
        # Save config
        config_path = output_dir / f"config_{idx}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print info for reference
        param_str = ", ".join([f"{k}={v}" for k, v in zip(param_names, values)])
        print(f"  [{idx:2d}] {param_str}")
        with open(config_path.parent / "README.txt", 'a') as f:
            f.write(f"config_{idx}.yaml: {param_str}\n")
    
    print(f"\n✓ Saved {len(combinations)} configs to {output_dir}")
    print(f"  Submit with: sbatch --array=0-{len(combinations)-1} submit_hyperparam_sweep.sh")


# Define hyperparameter grid - modify this for your sweep
# Phase 1: Fixed 100K dataset, vary LR, noise schedule, weight decay
PARAM_GRID = {
    "training.learning_rate": [5e-4, 2e-4, 1e-4],               # Learning rates
    "training.lr_scheduler.type": ["cosine", "linear"],         # Noise schedule types
    "training.weight_decay": [0.01, 0.1],                       # Weight decay values
}
# Total configs: 3 × 2 × 2 = 12 jobs

# Alternative sweep grids (uncomment to use):
# Phase 2: Vary noise schedule parameters for best LR
# PARAM_GRID = {
#     "training.noise_schedule.sigma_min": [0.0001, 0.001, 0.01],
#     "training.noise_schedule.sigma_max": [0.5, 1.0],
#     "training.noise_schedule.type": ["cosine", "linear"],
# }

# Phase 3: Fine-tune batch size and warmup
# PARAM_GRID = {
#     "training.batch_size": [64, 128, 256],
#     "training.warmup_epochs": [3, 5, 7],
# }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hyperparameter sweep configs")
    parser.add_argument("--base-config", default="configs/config_firstrun.yaml",
                       help="Base config file to use as template")
    parser.add_argument("--output-dir", default="configs/hyperparam_sweep",
                       help="Output directory for generated configs")
    parser.add_argument("--num-configs", type=int, default=None,
                       help="Limit number of configs generated (for testing)")
    
    args = parser.parse_args()
    
    generate_sweep_configs(
        args.base_config,
        PARAM_GRID,
        args.output_dir,
        args.num_configs
    )
