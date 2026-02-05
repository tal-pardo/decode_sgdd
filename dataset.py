"""
Simple dataset and forward operators for discrete diffusion.
"""

import torch
from torch.utils.data import Dataset


class BinarySequenceDataset(Dataset):
    """
    Simple binary sequence dataset for testing.
    Generates random binary sequences or loads from data.
    """
    
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=2, seed=42):
        """
        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length
            vocab_size: Vocabulary size (typically 2 for binary)
            seed: Random seed
        """
        torch.manual_seed(seed)
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].long()


class CustomBinaryDataset(Dataset):
    """Load custom binary dataset from file or tensor."""
    
    def __init__(self, data, seq_len=128, vocab_size=2):
        """
        Args:
            data: Tensor of shape (num_samples, seq_len) or file path
            seq_len: Expected sequence length
            vocab_size: Vocabulary size
        """
        if isinstance(data, str):
            # Load from file
            self.data = torch.load(data, weights_only=True)
        else:
            self.data = data
        
        # Validate
        assert self.data.shape[1] == seq_len, f"Expected seq_len={seq_len}, got {self.data.shape[1]}"
        assert self.data.min() >= 0 and self.data.max() < vocab_size, \
            f"Data must be in range [0, {vocab_size})"
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].long()


class ForwardOperator:
    """Base class for forward operators in inverse problems."""
    
    def __call__(self, x):
        """
        Forward operation: y = A(x)
        
        Args:
            x: Input sample(s)
        
        Returns:
            Measurement y
        """
        raise NotImplementedError
    
    def loss(self, x, y):
        """
        Measurement loss: ||A(x) - y||
        
        Args:
            x: Predicted sample(s)
            y: Observed measurement(s)
        
        Returns:
            Loss per sample
        """
        raise NotImplementedError
    
    def log_likelihood(self, x, y):
        """
        Log-likelihood: log p(y | x)
        
        Args:
            x: Sample(s)
            y: Measurement(s)
        
        Returns:
            Log-likelihood per sample
        """
        return -self.loss(x, y) / (self.sigma_noise ** 2)


class MaskingOperator(ForwardOperator):
    """
    Masking operator: observes only certain positions.
    y = M * x where M is binary mask
    """
    
    def __init__(self, mask, sigma_noise=0.01, device='cuda'):
        """
        Args:
            mask: Binary mask tensor, shape (seq_len,)
            sigma_noise: Noise level for likelihood
            device: Device
        """
        self.mask = mask.to(device).float()
        self.sigma_noise = sigma_noise
        self.device = device
    
    def __call__(self, x):
        """Apply masking."""
        return x * self.mask.unsqueeze(0)
    
    def loss(self, x, y):
        """L2 loss on masked positions."""
        predicted = self.__call__(x)
        return ((predicted - y) ** 2).sum(dim=-1)


class BinaryOperator(ForwardOperator):
    """
    Binary (XOR/AND) operator for Boolean functions.
    y = f(x_{i_0} op x_{i_1}) for pairs of positions
    """
    
    def __init__(self, operator_type='xor', num_pairs=None, seq_len=128, 
                 sigma_noise=0.01, seed=42, device='cuda'):
        """
        Args:
            operator_type: 'xor', 'and', 'or', 'nand'
            num_pairs: Number of position pairs to use
            seq_len: Sequence length
            sigma_noise: Noise level
            seed: Random seed for pair selection
            device: Device
        """
        torch.manual_seed(seed)
        
        self.operator_type = operator_type
        self.seq_len = seq_len
        self.sigma_noise = sigma_noise
        self.device = device
        
        # Randomly select pairs of positions
        if num_pairs is None:
            num_pairs = seq_len // 2
        
        self.num_pairs = num_pairs
        self.pairs = torch.randint(0, seq_len, (2, num_pairs))
    
    def __call__(self, x):
        """
        Apply binary operation.
        
        Args:
            x: Binary sequences, shape (batch, seq_len)
        
        Returns:
            Binary outputs, shape (batch, num_pairs)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x0 = x[:, self.pairs[0]]  # (batch, num_pairs)
        x1 = x[:, self.pairs[1]]  # (batch, num_pairs)
        
        if self.operator_type == 'xor':
            y = x0 ^ x1
        elif self.operator_type == 'and':
            y = x0 & x1
        elif self.operator_type == 'or':
            y = x0 | x1
        elif self.operator_type == 'nand':
            y = ~(x0 & x1)
        else:
            raise ValueError(f"Unknown operator: {self.operator_type}")
        
        return y.float().to(self.device)
    
    def loss(self, x, y):
        """
        Binary cross-entropy loss.
        
        Args:
            x: Predicted samples
            y: True outputs
        
        Returns:
            Loss per sample
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        pred = self.__call__(x).float()
        y = y.float()
        
        # BCEWithLogitsLoss expects floats in [0, 1]
        loss = -y * torch.log(pred + 1e-10) - (1 - y) * torch.log(1 - pred + 1e-10)
        return loss.mean(dim=-1)


class InpaintingOperator(ForwardOperator):
    """
    Inpainting operator: observe some positions, predict others.
    y = x * mask (observe positions where mask=1)
    """
    
    def __init__(self, mask_ratio=0.3, seq_len=128, sigma_noise=0.01, 
                 seed=42, device='cuda'):
        """
        Args:
            mask_ratio: Fraction of positions to observe
            seq_len: Sequence length
            sigma_noise: Noise level
            seed: Random seed
            device: Device
        """
        torch.manual_seed(seed)
        
        self.seq_len = seq_len
        self.sigma_noise = sigma_noise
        self.device = device
        
        # Create binary mask
        mask = torch.rand(seq_len) < mask_ratio
        self.mask = mask.float().to(device)
    
    def __call__(self, x):
        """Apply inpainting mask."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x * self.mask.unsqueeze(0)
    
    def loss(self, x, y):
        """
        L2 loss only on observed positions (mask=1).
        
        Args:
            x: Predicted samples
            y: Observed positions
        
        Returns:
            Loss per sample
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        predicted = self.__call__(x)
        observed_y = y * self.mask.unsqueeze(0)
        
        loss = ((predicted - observed_y) ** 2 * self.mask.unsqueeze(0)).sum(dim=-1)
        return loss
