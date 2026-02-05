"""
Minimal training loop for discrete diffusion models.
No distributed training, no complex logging - just PyTorch and prints.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Simple trainer for discrete diffusion models.
    """
    
    def __init__(self,
                 model,
                 graph,
                 noise_schedule,
                 train_loader,
                 val_loader=None,
                 learning_rate=1e-3,
                 num_epochs=10,
                 device='cuda',
                 save_dir='./checkpoints',
                 save_freq=1,
                 log_freq=10,
                 keep_strategy='all',
                 keep_last_n=3,
                 validation_freq=2,
                 num_samples_to_generate=100,
                 check_overfitting=True,
                 overfitting_sample_size=5000,
                 warmup_epochs=0,
                 lr_scheduler_type='none',
                 min_lr=1e-5):
        """
        Args:
            model: Score model to train
            graph: Diffusion graph (e.g., UniformGraph)
            noise_schedule: Noise schedule (e.g., CosineSchedule)
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to train on
            save_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N epochs
            log_freq: Log metrics every N batches
            keep_strategy: 'all' (save every), 'best_and_last' (only best+latest), 'keep_last_n' (keep last N)
            keep_last_n: When using 'keep_last_n' strategy, how many to keep
            validation_freq: Run sample validation every N epochs
            num_samples_to_generate: Number of samples to generate during validation
            check_overfitting: Enable/disable overfitting check
            overfitting_sample_size: Sample size from training data for overfitting check
        """
        self.model = model.to(device)
        self.graph = graph
        self.noise_schedule = noise_schedule
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.keep_strategy = keep_strategy
        self.keep_last_n = keep_last_n
        self.validation_freq = validation_freq
        self.num_samples_to_generate = num_samples_to_generate
        self.check_overfitting = check_overfitting
        self.overfitting_sample_size = overfitting_sample_size
        self.warmup_epochs = warmup_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.min_lr = min_lr
        
        # Try to load LDPC parity-check matrix from dataset
        self.H_matrix = self._extract_h_matrix()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'tensorboard'))
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        warmup_steps = len(train_loader) * warmup_epochs
        
        self.scheduler = self._create_scheduler(total_steps, warmup_steps, learning_rate, min_lr, lr_scheduler_type)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.saved_epochs = []  # Track saved epoch checkpoints for cleanup
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self._count_parameters():,}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            if self.val_loader is not None:
                val_loss = self._validate()
                self.val_losses.append(val_loss)
                
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                
                print(f"Epoch {epoch+1}/{self.num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {self.best_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} | "
                      f"Train Loss: {train_loss:.6f}")
                is_best = False
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if self.val_loader is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Run sample validation every validation_freq epochs
            if (epoch + 1) % self.validation_freq == 0:
                self._validate_samples()
            
            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(is_best=is_best if self.val_loader else False)
        
        self.writer.close()
        print("Training complete!")
        
        # Final validation on best model
        best_checkpoint_path = self.save_dir / 'checkpoint_best.pt'
        if best_checkpoint_path.exists():
            print("\n" + "="*60)
            print("FINAL VALIDATION ON BEST MODEL")
            print("="*60)
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self._validate_samples()
            print("="*60 + "\n")
        
        self._save_training_summary()
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, x_0 in enumerate(self.train_loader):
            x_0 = x_0.to(self.device)
            batch_size = x_0.shape[0]
            
            # Sample random time steps
            t = self.noise_schedule.sample_t(batch_size, device=self.device)
            sigma, dsigma_dt = self.noise_schedule(t)
            
            # Forward diffusion: q_sample
            # Sample x_t | x_0 from transition matrix
            x_t = self._q_sample(x_0, sigma)
            
            # Model prediction
            score_pred = self.model(x_t, sigma)
            
            # Loss: score entropy from graph
            loss = self.graph.score_entropy(score_pred, sigma, x_t, x_0)
            loss = loss.mean()  # Average over batch
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if (batch_idx + 1) % self.log_freq == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        return total_loss / num_batches
    
    def _validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_0 in self.val_loader:
                x_0 = x_0.to(self.device)
                batch_size = x_0.shape[0]
                
                # Sample random times
                t = self.noise_schedule.sample_t(batch_size, device=self.device)
                sigma, _ = self.noise_schedule(t)
                
                # Forward diffusion
                x_t = self._q_sample(x_0, sigma)
                
                # Model prediction
                score_pred = self.model(x_t, sigma)
                
                # Loss
                loss = self.graph.score_entropy(score_pred, sigma, x_t, x_0)
                total_loss += loss.mean().item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _q_sample(self, x_0, sigma):
        """
        Forward diffusion: sample x_t | x_0.
        Uses transition matrix from graph.
        
        Args:
            x_0: Clean samples, shape (batch, seq_len)
            sigma: Noise levels, shape (batch,)
        
        Returns:
            Noisy samples x_t, shape (batch, seq_len)
        """
        batch_size, seq_len = x_0.shape
        
        # Get transition probabilities from graph
        # Shape: (batch, seq_len, dim)
        transitions = self.graph.transition(x_0, sigma)
        
        # Sample from transition for each position
        x_t = torch.zeros_like(x_0)
        for i in range(seq_len):
            # Transition probabilities for position i
            trans_i = transitions[:, i, :]  # (batch, dim)
            
            # Sample using Gumbel-max trick
            gumbel = -torch.log(-torch.log(torch.rand_like(trans_i) + 1e-10) + 1e-10)
            x_t[:, i] = (trans_i.log() + gumbel).argmax(dim=-1)
        
        return x_t
    
    def _create_scheduler(self, total_steps, warmup_steps, base_lr, min_lr, scheduler_type):
        """
        Create learning rate scheduler with optional warmup.
        
        Args:
            total_steps: Total training steps
            warmup_steps: Number of warmup steps
            base_lr: Initial learning rate
            min_lr: Minimum learning rate (for cosine annealing)
            scheduler_type: 'cosine', 'linear', 'exponential', or 'none'
        
        Returns:
            scheduler object or None if scheduler_type='none'
        """
        if scheduler_type == 'none':
            return None
        
        # Create base scheduler
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr)
        elif scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=total_steps)
        elif scheduler_type == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            scheduler = ExponentialLR(self.optimizer, gamma=0.9999)
        else:
            return None
        
        # Add warmup if needed
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import ChainedScheduler, LinearLR
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
            scheduler = ChainedScheduler([warmup_scheduler, scheduler])
        
        return scheduler
    
    def _extract_h_matrix(self):
        """
        Extract LDPC parity-check matrix (H) from dataset.
        
        Returns:
            H matrix as numpy array
            
        Raises:
            RuntimeError if H_matrix not found in dataset
        """
        import numpy as np
        
        try:
            dataset = self.train_loader.dataset
            
            # Handle torch.utils.data.Subset wrapping
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            
            # Try to find stored H matrix in dataset
            if hasattr(dataset, 'H_matrix') and dataset.H_matrix is not None:
                return np.array(dataset.H_matrix, dtype=np.int32)
            else:
                # No fallback - must have saved H_matrix in dataset
                raise RuntimeError(
                    "H_matrix not found in dataset! "
                    "Please ensure the dataset was generated with ldpc_generate_dataset.py "
                    "which saves H_matrix alongside the codewords."
                )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract H_matrix from dataset: {e}. "
                f"Dataset must be loaded from a .pt file with H_matrix metadata."
            )
    
    def _is_valid_codeword(self, x):
        """
        Check if a binary sequence is a valid LDPC codeword.
        Valid iff H @ x = 0 (mod 2)
        
        Args:
            x: Binary sequence of length 128 (numpy array or torch tensor)
        
        Returns:
            True if valid codeword, False otherwise
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Check syndrome: H @ x should be 0 (mod 2)
        syndrome = (self.H_matrix @ x.astype(np.int32)) % 2
        return syndrome.sum() == 0
    
    def _count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _save_checkpoint(self, is_best=False):
        """Save model checkpoint with strategy-based cleanup."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Strategy: best_and_last - only keep best and latest
        if self.keep_strategy == 'best_and_last':
            # Save latest
            latest_path = self.save_dir / 'checkpoint_latest.pt'
            torch.save(checkpoint, latest_path)
            print(f"  Saved latest checkpoint: {latest_path}")
            
            # Save best
            if is_best:
                best_path = self.save_dir / 'checkpoint_best.pt'
                torch.save(checkpoint, best_path)
                print(f"  Saved best checkpoint: {best_path}")
        
        # Strategy: keep_last_n - keep only last N epoch checkpoints
        elif self.keep_strategy == 'keep_last_n':
            # Save latest
            latest_path = self.save_dir / 'checkpoint_latest.pt'
            torch.save(checkpoint, latest_path)
            print(f"  Saved latest checkpoint: {latest_path}")
            
            # Save epoch
            epoch_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
            self.saved_epochs.append(self.current_epoch)
            print(f"  Saved checkpoint: {epoch_path}")
            
            # Save best
            if is_best:
                best_path = self.save_dir / 'checkpoint_best.pt'
                torch.save(checkpoint, best_path)
                print(f"  Saved best checkpoint: {best_path}")
            
            # Clean up old epoch checkpoints
            if len(self.saved_epochs) > self.keep_last_n:
                old_epoch = self.saved_epochs.pop(0)
                old_path = self.save_dir / f'checkpoint_epoch_{old_epoch:03d}.pt'
                if old_path.exists():
                    old_path.unlink()
                    print(f"  Deleted old checkpoint: {old_path}")
        
        # Strategy: all - save everything (default)
        else:
            # Save latest
            latest_path = self.save_dir / 'checkpoint_latest.pt'
            torch.save(checkpoint, latest_path)
            
            # Save at epoch
            epoch_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
            
            # Save best
            if is_best:
                best_path = self.save_dir / 'checkpoint_best.pt'
                torch.save(checkpoint, best_path)
                print(f"  Saved best checkpoint: {best_path}")
            
            print(f"  Saved checkpoint: {epoch_path}")
    
    def _validate_samples(self):
        """Generate and validate samples to check model quality."""
        self.model.eval()
        print(f"\n  [Validation Epoch {self.current_epoch+1}] Generating {self.num_samples_to_generate} samples...")
        
        generated_samples = []
        with torch.no_grad():
            for _ in range(self.num_samples_to_generate):
                # Sample uniform noise
                x_t = torch.randint(0, self.graph.dim, (1, self.train_loader.dataset[0].shape[0]), device=self.device)
                
                # Reverse diffusion (sampling)
                sigma_schedule = np.linspace(1.0, 0.001, 50)
                for sigma in sigma_schedule:
                    sigma_t = torch.tensor([sigma], device=self.device)
                    score = self.model(x_t, sigma_t)
                    
                    # Simple sampling step (Euler)
                    dsigma = sigma_schedule[max(0, int(np.where(sigma_schedule == sigma)[0][0]) - 1)] - sigma
                    if dsigma != 0:
                        x_t = x_t + dsigma * score.argmax(dim=-1)
                        x_t = torch.clamp(x_t, 0, self.graph.dim - 1)
                        x_t = x_t.long()  # Ensure it stays as long tensor for embedding
                
                generated_samples.append(x_t.cpu())
        
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # 1. Uniqueness
        unique_samples = len(set([tuple(s.tolist()) for s in generated_samples]))
        uniqueness_pct = (unique_samples / len(generated_samples)) * 100
        
        # 2. Average Hamming Distance
        hamming_distances = []
        for i in range(len(generated_samples)):
            for j in range(i + 1, min(i + 10, len(generated_samples))):  # Compare with next 10
                dist = torch.sum(generated_samples[i] != generated_samples[j]).item()
                hamming_distances.append(dist)
        
        avg_hamming = np.mean(hamming_distances) if hamming_distances else 0
        
        # 3. LDPC Codeword Validity Check (H_matrix is guaranteed to exist)
        valid_codewords = 0
        for sample in generated_samples:
            if self._is_valid_codeword(sample):
                valid_codewords += 1
        valid_pct = (valid_codewords / len(generated_samples)) * 100
        
        # 4. Check if samples are in training set (overfitting check)
        in_training_pct = 0
        if self.train_loader.dataset is not None and self.check_overfitting:
            try:
                # Sample subset of training data for efficiency
                training_data = set()
                dataset_size = len(self.train_loader.dataset)
                sample_size = min(self.overfitting_sample_size, dataset_size)
                
                # Sample indices randomly
                indices = torch.randperm(dataset_size)[:sample_size]
                
                # Collect sampled training data
                for idx in indices:
                    sample = self.train_loader.dataset[idx]
                    training_data.add(tuple(sample.tolist()))
                
                in_training = sum(1 for s in generated_samples if tuple(s.tolist()) in training_data)
                in_training_pct = (in_training / len(generated_samples)) * 100
            except:
                in_training_pct = 0
        else:
            in_training_pct = 0
        
        # Log metrics
        self.writer.add_scalar('Validation/uniqueness_pct', uniqueness_pct, self.current_epoch)
        self.writer.add_scalar('Validation/avg_hamming_distance', avg_hamming, self.current_epoch)
        if self.H_matrix is not None:
            self.writer.add_scalar('Validation/valid_codewords_pct', valid_pct, self.current_epoch)
        self.writer.add_scalar('Validation/in_training_pct', in_training_pct, self.current_epoch)
        
        # Print results
        print(f"  Unique samples: {unique_samples}/{len(generated_samples)} ({uniqueness_pct:.1f}%)")
        print(f"  Avg Hamming distance: {avg_hamming:.1f}")
        if self.H_matrix is not None:
            print(f"  Valid LDPC codewords: {valid_codewords}/{len(generated_samples)} ({valid_pct:.1f}%)")
        print(f"  In training set: {in_training_pct:.1f}%")
        print()
        
        self.model.train()
    
    def _save_training_summary(self):
        """Save training summary to JSON."""
        summary = {
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': float(self.best_loss),
            'timestamp': datetime.now().isoformat(),
        }
        
        summary_path = self.save_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved training summary: {summary_path}")
    
    def load_checkpoint(self, path):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_model(self, path):
        """Save model for inference."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")
    
    def load_model(self, path):
        """Load model for inference."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")
