"""
Inference script for sampling from trained discrete diffusion models.
"""

import torch
import argparse
from pathlib import Path

from model import create_score_model
from graph import UniformGraph
from noise_schedule import get_schedule
from sampler import UnconditionalSampler, SplitGibbsSampler
from dataset import BinarySequenceDataset, BinaryOperator


def sample_unconditional(args):
    """Sample unconditional samples from trained model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Unconditional Sampling")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load model
    model = create_score_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded model from {args.model_path}")
    
    # Setup graph and schedule
    graph = UniformGraph(dim=args.vocab_size)
    noise_schedule = get_schedule(
        schedule_type=args.noise_schedule,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max
    )
    
    # Create sampler
    sampler = UnconditionalSampler(
        model=model,
        graph=graph,
        noise_schedule=noise_schedule,
        device=device,
        num_steps=args.num_steps
    )
    print(f"✓ Created {sampler.__class__.__name__} with {args.num_steps} steps")
    
    # Sample
    print(f"\nSampling {args.num_samples} samples...")
    samples = sampler.sample(num_samples=args.num_samples, verbose=True)
    
    # Save samples
    output_path = Path(args.output_dir) / 'samples.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_path)
    print(f"✓ Saved samples to {output_path}")
    
    # Print statistics
    print(f"\nSample Statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Min: {samples.min().item()}, Max: {samples.max().item()}")
    for idx in range(min(5, len(samples))):
        print(f"  Sample {idx}: {samples[idx][:20].tolist()}...")
    
    return samples


def sample_conditional(args):
    """Sample conditional samples using Split Gibbs sampler."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Conditional Sampling (Split Gibbs)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load model
    model = create_score_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded model from {args.model_path}")
    
    # Setup graph and schedule
    graph = UniformGraph(dim=args.vocab_size)
    noise_schedule = get_schedule(
        schedule_type=args.noise_schedule,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max
    )
    
    # Create forward operator (e.g., binary operation)
    forward_op = BinaryOperator(
        operator_type=args.operator_type,
        num_pairs=args.num_pairs,
        seq_len=args.seq_len,
        device=device
    )
    print(f"✓ Created {args.operator_type.upper()} forward operator")
    
    # Generate ground truth and observation
    print(f"\nGenerating ground truth samples...")
    gt_dataset = BinarySequenceDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed
    )
    gt_samples = torch.stack([gt_dataset[i] for i in range(args.num_samples)])
    
    # Generate observations
    observations = forward_op(gt_samples.to(device))
    print(f"✓ Generated {len(observations)} observations")
    
    # Create sampler
    sampler = SplitGibbsSampler(
        model=model,
        graph=graph,
        noise_schedule=noise_schedule,
        forward_op=forward_op,
        device=device,
        num_steps=args.num_steps,
        mh_steps=args.mh_steps,
        alpha=args.alpha
    )
    print(f"✓ Created {sampler.__class__.__name__}")
    print(f"  - Diffusion steps: {args.num_steps}")
    print(f"  - MH steps: {args.mh_steps}")
    print(f"  - Alpha: {args.alpha}")
    
    # Sample for each observation
    print(f"\nSampling predictions...")
    predictions = []
    for i in range(min(args.num_samples, 5)):  # Sample for first 5
        obs = observations[i:i+1]
        pred = sampler.sample(num_samples=1, observation=obs, verbose=True)
        predictions.append(pred)
        print(f"  Observation {i}: {obs.tolist()}")
        print(f"  Prediction {i}: {pred.tolist()}")
    
    predictions = torch.cat(predictions, dim=0)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(gt_samples, output_dir / 'gt_samples.pt')
    torch.save(observations, output_dir / 'observations.pt')
    torch.save(predictions, output_dir / 'predictions.pt')
    
    print(f"\n✓ Saved results to {output_dir}")
    
    return gt_samples, observations, predictions


def main():
    """Parse arguments and run sampling."""
    parser = argparse.ArgumentParser(description='Sample from trained discrete diffusion model')
    
    # Mode
    parser.add_argument('--mode', choices=['unconditional', 'conditional'],
                        default='unconditional', help='Sampling mode')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model-type', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--vocab-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    
    # Sampling arguments
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--noise-schedule', choices=['cosine', 'linear', 'exponential'],
                        default='cosine')
    parser.add_argument('--sigma-min', type=float, default=0.001)
    parser.add_argument('--sigma-max', type=float, default=1.0)
    
    # Conditional sampling arguments
    parser.add_argument('--mh-steps', type=int, default=50,
                        help='Metropolis-Hastings steps (conditional only)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Likelihood weight (conditional only)')
    parser.add_argument('--operator-type', choices=['xor', 'and', 'or', 'nand'],
                        default='xor', help='Binary operator (conditional only)')
    parser.add_argument('--num-pairs', type=int, default=64,
                        help='Number of operator pairs (conditional only)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./samples',
                        help='Output directory for samples')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Run sampling
    if args.mode == 'unconditional':
        sample_unconditional(args)
    elif args.mode == 'conditional':
        sample_conditional(args)


if __name__ == '__main__':
    main()
