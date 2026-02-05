"""
Configuration loader for SEDD training.
Supports YAML config files with command-line overrides.
"""

import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    path: str
    num_samples: Optional[int]
    seq_len: int
    vocab_size: int
    val_split: float
    seed: int


@dataclass
class ModelConfig:
    type: str
    hidden_dim: int
    n_heads: int
    n_blocks: int
    mlp_ratio: float
    dropout: float
    scale_by_sigma: bool
    embedding_dim: int
    num_layers: int


@dataclass
class NoiseScheduleConfig:
    type: str
    sigma_min: float
    sigma_max: float


@dataclass
class LRSchedulerConfig:
    type: str
    warmup_epochs: int
    min_lr: float


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    clip_grad_norm: float
    lr_scheduler: LRSchedulerConfig


@dataclass
class CheckpointingConfig:
    save_dir: str
    save_freq: int
    save_best: bool
    keep_strategy: str  # 'all', 'best_and_last', 'keep_last_n'
    keep_last_n: int


@dataclass
class LoggingConfig:
    log_freq: int
    print_model_summary: bool
    log_attention: bool
    validation_freq: int  # Generate samples every N epochs
    num_samples_to_generate: int  # Number of samples to generate during validation
    check_overfitting: bool  # Enable overfitting check
    overfitting_sample_size: int  # Sample size for overfitting check
    check_overfitting: bool  # Check if generated samples are in training set
    overfitting_sample_size: int  # Sample size from training data for overfitting check


@dataclass
class DeviceConfig:
    device: str
    mixed_precision: bool


@dataclass
class DebugConfig:
    enabled: bool
    check_nan: bool
    log_gradients: bool
    visualize_attention: bool
    test_forward_pass: bool


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    noise_schedule: NoiseScheduleConfig
    training: TrainingConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    device: DeviceConfig
    debug: DebugConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Config object with all parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse nested configs
    dataset = DatasetConfig(**config_dict['dataset'])
    model = ModelConfig(**config_dict['model'])
    noise_schedule = NoiseScheduleConfig(**config_dict['noise_schedule'])
    
    lr_scheduler_dict = config_dict['training'].pop('lr_scheduler')
    lr_scheduler = LRSchedulerConfig(**lr_scheduler_dict)
    training = TrainingConfig(**config_dict['training'], lr_scheduler=lr_scheduler)
    
    checkpointing = CheckpointingConfig(**config_dict['checkpointing'])
    logging = LoggingConfig(**config_dict['logging'])
    device = DeviceConfig(**config_dict['device'])
    debug = DebugConfig(**config_dict['debug'])
    
    return Config(
        dataset=dataset,
        model=model,
        noise_schedule=noise_schedule,
        training=training,
        checkpointing=checkpointing,
        logging=logging,
        device=device,
        debug=debug
    )


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path where to save the config
    """
    config_dict = {
        'dataset': config.dataset.__dict__,
        'model': config.model.__dict__,
        'noise_schedule': config.noise_schedule.__dict__,
        'training': {
            **{k: v for k, v in config.training.__dict__.items() if k != 'lr_scheduler'},
            'lr_scheduler': config.training.lr_scheduler.__dict__
        },
        'checkpointing': config.checkpointing.__dict__,
        'logging': config.logging.__dict__,
        'device': config.device.__dict__,
        'debug': config.debug.__dict__,
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"✓ Config saved to: {save_path}")


def print_config(config: Config) -> None:
    """Pretty print configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    print("\n[DATASET]")
    print(f"  Path: {config.dataset.path}")
    print(f"  Samples: {config.dataset.num_samples or 'all from file'}")
    print(f"  Seq length: {config.dataset.seq_len}")
    print(f"  Vocab size: {config.dataset.vocab_size}")
    
    print("\n[MODEL]")
    print(f"  Type: {config.model.type}")
    if config.model.type == 'transformer':
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  Attention heads: {config.model.n_heads}")
        print(f"  Blocks: {config.model.n_blocks}")
        print(f"  MLP ratio: {config.model.mlp_ratio}")
    else:
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  Layers: {config.model.num_layers}")
    print(f"  Dropout: {config.model.dropout}")
    print(f"  Scale by sigma: {config.model.scale_by_sigma}")
    
    print("\n[NOISE SCHEDULE]")
    print(f"  Type: {config.noise_schedule.type}")
    print(f"  Sigma min: {config.noise_schedule.sigma_min}")
    print(f"  Sigma max: {config.noise_schedule.sigma_max}")
    
    print("\n[TRAINING]")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate:.2e}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Weight decay: {config.training.weight_decay:.2e}")
    print(f"  Gradient clip: {config.training.clip_grad_norm}")
    print(f"  LR scheduler: {config.training.lr_scheduler.type}")
    
    print("\n[CHECKPOINTING]")
    print(f"  Save dir: {config.checkpointing.save_dir}")
    print(f"  Save freq: every {config.checkpointing.save_freq} epoch(s)")
    print(f"  Keep strategy: {config.checkpointing.keep_strategy}")
    if config.checkpointing.keep_strategy == 'keep_last_n':
        print(f"  Keep last N: {config.checkpointing.keep_last_n}")
    
    print("\n[DEVICE]")
    print(f"  Device: {config.device.device}")
    print(f"  Mixed precision: {config.device.mixed_precision}")
    
    if config.debug.enabled:
        print("\n[DEBUG]")
        print(f"  Check NaN: {config.debug.check_nan}")
        print(f"  Log gradients: {config.debug.log_gradients}")
        print(f"  Test forward: {config.debug.test_forward_pass}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test loading config
    config = load_config('config_debug.yaml')  # Run from configs/ directory
    print_config(config)
