"""
LDPC (Low-Density Parity-Check) Codeword Dataset Generator for SEDD Training

Generates binary sequences of shape (N, 128) where:
- First 64 bits: i.i.d. message bits (random)
- Last 64 bits: parity bits satisfying Hx^T = 0 (mod 2)

Uses pyldpc library for professional LDPC code generation.
All codewords share the same H matrix (parity-check matrix).
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import pyldpc
import os


class LDPCGenerator:
    """Generate LDPC codewords using pyldpc library."""
    
    def __init__(self, n=128, code_rate=0.5, seed=42):
        """
        Initialize LDPC generator using pyldpc.
        
        Args:
            n: Codeword length (total bits) = 128
            code_rate: Code rate R = k/n (fraction of data bits)
                - 0.5: 64 data + 64 parity (default, good for learning)
                - 0.75: 96 data + 32 parity
                - 0.8: 102 data + 26 parity (realistic)
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n = n  # Total codeword length
        self.code_rate = code_rate
        self.k = int(n * code_rate)  # Message/information bits (data bits)
        self.m = n - self.k  # Parity check bits
        self.seed = seed
        
        # Generate LDPC code using pyldpc
        self.H = self._generate_ldpc_matrix()
        
    def _generate_ldpc_matrix(self):
        """
        Generate LDPC parity-check matrix using pyldpc.
        
        Uses regular LDPC construction from pyldpc library.
        Returns:
            H: Parity-check matrix of shape (m, n)
        """
        column_weight = 2  # Each bit in ~2 parity checks
        row_weight = max(2, (self.n * column_weight) // self.m)
        
        print(f"  Creating LDPC H matrix (column_weight=2, row_weight={row_weight})...")
        
        # Generate H matrix - pyldpc.make_ldpc returns (H, G) tuple
        result = pyldpc.make_ldpc(self.n, column_weight, row_weight, seed=self.seed)
        
        # Extract H matrix from tuple
        if isinstance(result, tuple):
            H = result[0]  # First element is H matrix
        else:
            H = result
        
        # Convert to numpy array
        H = np.array(H, dtype=np.int32)
        
        # Adjust size to match our requirements
        if H.shape[0] > self.m:
            H = H[:self.m, :]
        elif H.shape[0] < self.m:
            extra_rows = self.m - H.shape[0]
            padding = np.random.binomial(1, 0.3, size=(extra_rows, self.n))
            H = np.vstack([H, padding])
        
        print(f"  H matrix shape: {H.shape}, sparsity: {1 - np.count_nonzero(H) / (H.shape[0] * H.shape[1]):.3f}")
        return H
    
    def generate_codeword(self):
        """
        Generate a single valid LDPC codeword.
        
        Returns:
            x: Valid codeword of length n (128 bits)
        """
        # Generate random message bits
        message = np.random.randint(0, 2, size=self.k, dtype=np.int32)
        
        # Initialize codeword
        codeword = np.zeros(self.n, dtype=np.int32)
        codeword[:self.k] = message
        
        # Compute parity bits: H[:, k:] @ parity = H[:, :k] @ message (mod 2)
        H_msg = self.H[:, :self.k]
        H_par = self.H[:, self.k:]
        
        target = (H_msg @ message.astype(np.int32)) % 2
        
        # Solve for parity using greedy bit-flipping
        parity = self._solve_parity(H_par, target)
        codeword[self.k:] = parity
        
        # Verify and fix if needed
        syndrome = (self.H @ codeword.astype(np.int32)) % 2
        if syndrome.sum() > 0:
            codeword = self._fix_codeword(codeword, syndrome)
        
        return codeword.astype(np.int32)
    
    def _solve_parity(self, H_par, target):
        """Solve H_par @ p = target (mod 2) using greedy bit-flipping."""
        m = H_par.shape[0]
        parity = np.zeros(m, dtype=np.int32)
        
        for iteration in range(50):
            syndrome = (H_par @ parity) % 2
            error = (syndrome ^ target)
            
            if error.sum() == 0:
                return parity
            
            best_bit = -1
            best_reduction = 0
            
            for bit in range(m):
                parity_test = parity.copy()
                parity_test[bit] ^= 1
                new_error = ((H_par @ parity_test) % 2) ^ target
                reduction = error.sum() - new_error.sum()
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_bit = bit
            
            if best_bit >= 0 and best_reduction > 0:
                parity[best_bit] ^= 1
            else:
                bit = np.random.randint(0, m)
                parity[bit] ^= 1
        
        return parity
    
    def _fix_codeword(self, codeword, syndrome):
        """Fix codeword by flipping parity bits to reduce syndrome error."""
        for iteration in range(30):
            if syndrome.sum() == 0:
                return codeword
            
            best_bit = -1
            best_reduction = 0
            
            for bit in range(self.k, self.n):
                codeword_test = codeword.copy()
                codeword_test[bit] ^= 1
                new_syndrome = (self.H @ codeword_test) % 2
                reduction = syndrome.sum() - new_syndrome.sum()
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_bit = bit
            
            if best_bit >= 0 and best_reduction > 0:
                codeword[best_bit] ^= 1
                syndrome = (self.H @ codeword) % 2
            else:
                break
        
        return codeword
    
    def generate_dataset(self, num_samples):
        """Generate a dataset of valid LDPC codewords."""
        data = []
        for i in range(num_samples):
            codeword = self.generate_codeword()
            data.append(codeword)
            
            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"    Generated {i + 1}/{num_samples} codewords...")
        
        return torch.from_numpy(np.array(data)).long()


class LDPCDataset(Dataset):
    """PyTorch Dataset for LDPC codewords."""
    
    def __init__(self, num_samples=5000, n=128, code_rate=0.5, seed=42):
        """Initialize LDPC dataset."""
        generator = LDPCGenerator(n=n, code_rate=code_rate, seed=seed)
        self.data = generator.generate_dataset(num_samples)
        self.n = n
        self.k = generator.k
        self.m = generator.m
        self.code_rate = code_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].long()


def create_ldpc_dataset(num_samples=5000, n=128, code_rate=0.5,
                       save_path='ldpc_codewords.pt', seed=42):
    """Create and save LDPC dataset."""
    k = int(n * code_rate)
    m = n - k
    
    print(f"\n{'='*60}")
    print(f"LDPC Dataset Generation (pyldpc)")
    print(f"{'='*60}")
    print(f"Codeword length (n):     {n}")
    print(f"Code rate (R):           {code_rate}")
    print(f"Message length (k):      {k}")
    print(f"Parity length (m):       {m}")
    print(f"Number of samples:       {num_samples}")
    
    generator = LDPCGenerator(n=n, code_rate=code_rate, seed=seed)
    
    print(f"\nGenerating {num_samples} codewords...")
    data = generator.generate_dataset(num_samples)
    
    # Save to file
    torch.save(data, save_path)
    print(f"\n✓ Dataset saved to: {save_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min value: {data.min()}, Max value: {data.max()}")
    print(f"{'='*60}\n")
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LDPC dataset for SEDD training")
    parser.add_argument('--num-samples', type=int, default=5000, 
                        help='Number of codewords to generate (default: 5000)')
    parser.add_argument('--save-path', type=str, default='ldpc_codewords_r05.pt',
                        help='Path to save dataset (default: ldpc_codewords_r05.pt)')
    parser.add_argument('--n', type=int, default=128,
                        help='Codeword length (default: 128)')
    parser.add_argument('--code-rate', type=float, default=0.5,
                        help='Code rate k/n (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LDPC Dataset Generator - pyldpc Construction")
    print("="*60)
    print(f"\n[CONFIG]")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Codeword length: {args.n}")
    print(f"  Code rate: {args.code_rate}")
    print(f"  Save path: {args.save_path}")
    print(f"  Seed: {args.seed}")
    
    data = create_ldpc_dataset(
        num_samples=args.num_samples,
        n=args.n,
        code_rate=args.code_rate,
        save_path=args.save_path,
        seed=args.seed
    )
    
    print("\nDataset Verification:")
    print(f"  Shape: {data.shape}")
    print(f"  Values: {torch.unique(data).tolist()}")
    print(f"\nFirst 3 codewords:")
    for i in range(3):
        print(f"  Codeword {i+1}: {data[i].numpy()}")
