#!/usr/bin/env python3
"""
SEDD Transformer vs MLP Comparison Script

Train both models on LDPC dataset and compare:
- Loss convergence
- Training time
- Model capacity
- Performance on validation set
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train import train_binary_diffusion
import argparse


def compare_models():
    """Run training comparison between Transformer and MLP."""
    
    parser = argparse.ArgumentParser(description='Compare SEDD Transformer vs MLP on LDPC data')
    
    # Shared arguments
    parser.add_argument('--dataset-path', type=str, default='ldpc_codewords_r05_100k.pt',
                        help='Path to LDPC dataset')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--vocab-size', type=int, default=2,
                        help='Vocabulary size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir-base', type=str, default='./checkpoints_comparison',
                        help='Base directory for checkpoints')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SEDD TRANSFORMER vs MLP COMPARISON ON LDPC DATA")
    print("="*80)
    
    # Model configurations
    configs = {
        'transformer': {
            'model_type': 'transformer',
            'hidden_dim': 256,
            'num_layers': 6,
            'n_heads': 8,
            'embedding_dim': None,  # Not used for transformer
            'save_dir': f'{args.save_dir_base}/transformer',
            'description': 'SEDD Transformer (6 blocks, 8 heads, 256 hidden)'
        },
        'mlp': {
            'model_type': 'mlp',
            'hidden_dim': 256,
            'num_layers': 4,
            'embedding_dim': 128,
            'n_heads': None,  # Not used for MLP
            'save_dir': f'{args.save_dir_base}/mlp',
            'description': 'MLP Baseline (4 layers, 128 embedding, 256 hidden)'
        }
    }
    
    results = {}
    
    for model_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}: {config['description']}")
        print(f"{'='*80}")
        
        # Create model-specific args
        model_args = argparse.Namespace(
            dataset_path=args.dataset_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            model_type=config['model_type'],
            embedding_dim=config['embedding_dim'] or 64,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            n_heads=config.get('n_heads', 8),
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            num_samples=10000,  # Ignored (using dataset-path)
            val_split=0.1,
            noise_schedule='cosine',
            sigma_min=0.001,
            sigma_max=1.0,
            save_dir=config['save_dir'],
            save_freq=1,
            log_freq=50,
            seed=args.seed
        )
        
        # Train
        try:
            trainer = train_binary_diffusion(model_args)
            results[model_name] = {
                'trainer': trainer,
                'best_loss': min(trainer.val_losses) if trainer.val_losses else float('inf'),
                'final_loss': trainer.train_losses[-1] if trainer.train_losses else float('inf'),
                'epochs': len(trainer.train_losses),
                'params': sum(p.numel() for p in trainer.model.parameters())
            }
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    comparison_table = []
    for model_name, result in results.items():
        comparison_table.append({
            'Model': model_name.upper(),
            'Parameters': f"{result['params']:,}",
            'Best Val Loss': f"{result['best_loss']:.4f}",
            'Final Train Loss': f"{result['final_loss']:.4f}",
            'Epochs': result['epochs']
        })
    
    if comparison_table:
        print(f"{'Model':<15} {'Parameters':<15} {'Best Val Loss':<15} {'Final Train Loss':<18} {'Epochs':<10}")
        print("-" * 73)
        for row in comparison_table:
            print(f"{row['Model']:<15} {row['Parameters']:<15} {row['Best Val Loss']:<15} {row['Final Train Loss']:<18} {row['Epochs']:<10}")
    
    print(f"\n{'='*80}")
    
    if len(results) == 2:
        trans_best = results.get('transformer', {}).get('best_loss', float('inf'))
        mlp_best = results.get('mlp', {}).get('best_loss', float('inf'))
        
        if trans_best < float('inf') and mlp_best < float('inf'):
            diff = abs(trans_best - mlp_best)
            winner = 'Transformer' if trans_best < mlp_best else 'MLP'
            print(f"\n🏆 WINNER: {winner} (Best Loss Difference: {diff:.4f})")
    
    print(f"\nCheckpoints saved to:")
    for model_name, result in results.items():
        print(f"  - {model_name}: {configs[model_name]['save_dir']}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    compare_models()
