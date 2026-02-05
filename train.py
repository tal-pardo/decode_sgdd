"""
Example training script for discrete diffusion on binary sequences.
Single GPU, minimal dependencies - Windows compatible.
Uses YAML config files for easy parameter management.
"""

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path
import sys

# Import from local modules
from model import create_score_model
from transformer_model import create_sedd_transformer
from graph import UniformGraph
from noise_schedule import get_schedule
from dataset import BinarySequenceDataset, CustomBinaryDataset
from trainer import Trainer
from configs.config_loader import load_config, print_config, save_config


def train_binary_diffusion(config):
    """Train discrete diffusion model on binary sequences using config."""
    
    # Setup
    device = torch.device(config.device.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Discrete Diffusion Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {config.model.type.upper()}")
    print(f"Dataset: {config.dataset.path}")
    print(f"Sequence length: {config.dataset.seq_len}")
    print(f"Vocabulary size: {config.dataset.vocab_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"{'='*60}\n")
    
    # Debug test forward pass if enabled
    if config.debug.test_forward_pass:
        print("Running debug forward pass test...")
        test_model = create_sedd_transformer(
            vocab_size=config.dataset.vocab_size,
            seq_len=config.dataset.seq_len,
            hidden_size=config.model.hidden_dim,
            n_heads=config.model.n_heads,
            n_blocks=config.model.n_blocks,
            dropout=config.model.dropout,
            scale_by_sigma=config.model.scale_by_sigma,
            device=device
        )
        test_x = torch.randint(0, config.dataset.vocab_size, (2, config.dataset.seq_len), device=device)
        test_sigma = torch.rand(2, device=device)
        with torch.no_grad():
            test_out = test_model(test_x, test_sigma)
        assert test_out.shape == (2, config.dataset.seq_len, config.dataset.vocab_size)
        print(f"[OK] Forward pass test passed: input {test_x.shape} -> output {test_out.shape}\n")
        del test_model
    
    # Create graph (use UniformGraph for binary)
    graph = UniformGraph(dim=config.dataset.vocab_size)
    print(f"[OK] Created {graph.__class__.__name__} with dim={graph.dim}")
    
    # Create noise schedule
    noise_schedule = get_schedule(
        schedule_type=config.noise_schedule.type,
        sigma_min=config.noise_schedule.sigma_min,
        sigma_max=config.noise_schedule.sigma_max
    )
    print(f"[OK] Created {config.noise_schedule.type.capitalize()} noise schedule")
    
    # Create model
    if config.model.type.lower() == 'transformer':
        # SEDD Transformer
        model = create_sedd_transformer(
            vocab_size=config.dataset.vocab_size,
            seq_len=config.dataset.seq_len,
            hidden_size=config.model.hidden_dim,
            n_heads=config.model.n_heads,
            n_blocks=config.model.n_blocks,
            dropout=config.model.dropout,
            scale_by_sigma=config.model.scale_by_sigma,
            device=device
        )
    else:
        # MLP or other models
        model = create_score_model(
            model_type=config.model.type,
            vocab_size=config.dataset.vocab_size,
            seq_len=config.dataset.seq_len,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Created {config.model.type.upper()} model ({total_params:,} parameters)")
    
    # Create dataset
    if config.dataset.path:
        # Load custom LDPC dataset from file
        print(f"Loading dataset from: {config.dataset.path}")
        dataset = CustomBinaryDataset(
            config.dataset.path,
            seq_len=config.dataset.seq_len,
            vocab_size=config.dataset.vocab_size
        )
        
        # Use subset if num_samples specified
        if config.dataset.num_samples is not None:
            total_samples = min(config.dataset.num_samples, len(dataset))
            indices = torch.randperm(len(dataset))[:total_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        print(f"[OK] Loaded dataset with {len(dataset)} samples from {config.dataset.path}")
    else:
        # Generate random binary sequences
        dataset = BinarySequenceDataset(
            num_samples=config.dataset.num_samples or 10000,
            seq_len=config.dataset.seq_len,
            vocab_size=config.dataset.vocab_size,
            seed=config.dataset.seed
        )
        print(f"[OK] Generated random binary dataset with {len(dataset)} samples")
    
    # Train/val split
    val_size = int(len(dataset) * config.dataset.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.dataset.seed)
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"[OK] Created data loaders (train: {len(train_dataset)}, val: {len(val_dataset)})")
    
    # Create checkpoint directory
    Path(config.checkpointing.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory
    config_save_path = Path(config.checkpointing.save_dir) / 'config.yaml'
    save_config(config, str(config_save_path))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        graph=graph,
        noise_schedule=noise_schedule,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        device=device,
        save_dir=config.checkpointing.save_dir,
        save_freq=config.checkpointing.save_freq,
        log_freq=config.logging.log_freq,
        keep_strategy=config.checkpointing.keep_strategy,
        keep_last_n=config.checkpointing.keep_last_n,
        validation_freq=config.logging.validation_freq,
        num_samples_to_generate=config.logging.num_samples_to_generate,
        check_overfitting=config.logging.check_overfitting,
        overfitting_sample_size=config.logging.overfitting_sample_size,
        warmup_epochs=config.training.lr_scheduler.warmup_epochs,
        lr_scheduler_type=config.training.lr_scheduler.type,
        min_lr=config.training.lr_scheduler.min_lr
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_model_path = Path(config.checkpointing.save_dir) / 'model_final.pt'
    trainer.save_model(str(final_model_path))
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Checkpoints saved to: {config.checkpointing.save_dir}")
    print(f"{'='*60}\n")
    
    return trainer


def main():
    """Parse arguments and run training from config file."""
    parser = argparse.ArgumentParser(
        description='Train discrete diffusion model from YAML config'
    )
    
    parser.add_argument('--config', type=str, default='configs/config_debug.yaml',
                        help='Path to YAML config file (default: configs/config_debug.yaml)')
    parser.add_argument('--override-epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--override-lr', type=float, default=None,
                        help='Override learning_rate from config')
    parser.add_argument('--override-batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--override-model-type', type=str, choices=['mlp', 'transformer'], default=None,
                        help='Override model type from config')
    
    args = parser.parse_args()
    
    # Resolve config path relative to workspace root if needed
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Assume relative to workspace root
        config_path = Path.cwd() / config_path
    
    # Load config from YAML
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Apply command-line overrides
    if args.override_epochs is not None:
        config.training.num_epochs = args.override_epochs
    if args.override_lr is not None:
        config.training.learning_rate = args.override_lr
    if args.override_batch_size is not None:
        config.training.batch_size = args.override_batch_size
    if args.override_model_type is not None:
        config.model.type = args.override_model_type
    
    # Print config
    print_config(config)
    
    # Set seed
    torch.manual_seed(config.dataset.seed)
    
    # Train
    train_binary_diffusion(config)


if __name__ == '__main__':
    main()
