"""
LDPC (Low-Density Parity-Check) Codeword Dataset Generator for SEDD Training

Generates binary sequences of shape (N, 128) where:
- First 64 bits: i.i.d. message bits (random)
- Last 64 bits: parity bits satisfying Hx^T = 0 (mod 2)

Uses pyldpc library for professional LDPC code generation with proper encoding.
All codewords share the same VALID H matrix (parity-check matrix).
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pyldpc import make_ldpc
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
        self.seed = seed
        
        # Generate LDPC code using pyldpc - returns both H and G!
        # The actual k will be determined by what pyldpc returns
        self.H, self.G = self._generate_ldpc_matrices()
        
        # Now set k and m based on actual G matrix size
        self.k = self.G.shape[1]  # Number of columns in G = message bits
        self.m = self.H.shape[0]  # Number of rows in H = parity bits
        
    def _generate_ldpc_matrices(self):
        """
        Generate LDPC parity-check matrix (H) and generator matrix (G) using pyldpc.
        
        Uses systematic LDPC construction from pyldpc library.
        Key constraint: n must be a multiple of d_c (row_weight)
        For n=128: valid d_c values are 2, 4, 8, 16, 32, 64, 128
        
        Returns:
            H: Parity-check matrix of shape (m, n)
            G: Generator matrix for encoding (k, n)
        """
        # Choose d_v and d_c such that n is a multiple of d_c
        # For exact 0.5 rate (64 data, 64 parity) with n=128:
        # Use d_v=2 (column weight), d_c=4 (row weight)
        d_v = 2  # Column weight (number of 1s per column) 
        d_c = 4  # Row weight (number of 1s per row) - 128 % 4 = 0 ✓
        
        print(f"  Creating LDPC matrices:")
        print(f"    n={self.n}, d_v={d_v}, d_c={d_c}")
        print(f"    Constraint: n % d_c = {self.n} % {d_c} = {self.n % d_c} (must be 0)")
        
        if self.n % d_c != 0:
            print(f"    ⚠ WARNING: n not multiple of d_c! This may cause shape issues.")
        
        # Generate both H and G matrices with systematic form
        # pyldpc will determine k based on the code structure
        H, G = make_ldpc(
            self.n, 
            d_v,      # Column weight
            d_c,      # Row weight  
            seed=self.seed,
            systematic=True,
            sparse=False
        )
        
        # Convert to numpy arrays
        H = np.array(H, dtype=np.int32)
        G = np.array(G, dtype=np.int32)
        
        # Report actual dimensions
        k_actual = G.shape[0]
        m_actual = H.shape[0]
        rate_actual = k_actual / self.n if self.n > 0 else 0
        
        print(f"  ✓ Generated matrices:")
        print(f"    H: {H.shape} (parity checks: {m_actual}, codeword length: {self.n})")
        print(f"    G: {G.shape} (message bits: {k_actual}, codeword length: {self.n})")
        print(f"    Actual code rate: {k_actual}/{self.n} = {rate_actual:.3f}")
        print(f"    H sparsity: {1 - np.count_nonzero(H) / (H.shape[0] * H.shape[1]):.3f}")
        
        return H, G
    
    def generate_codeword(self):
        """
        Generate a single valid LDPC codeword using direct matrix multiplication.
        
        For systematic LDPC: c = message @ G.T (mod 2)
        This guarantees H @ c = 0 (mod 2) by construction.
        
        Returns:
            x: Valid codeword of length n (128 bits)
        """
        # Generate random message bits (k bits)
        message = np.random.randint(0, 2, size=self.k, dtype=np.int32)
        
        # Compute codeword: c = message @ G.T (mod 2)
        # G is (n, k), so G.T is (k, n)
        # message is (k,), so message @ G.T = (n,)
        # This guarantees a valid codeword: H @ c = 0 (mod 2)
        codeword = (message @ self.G.T) % 2
        codeword = np.array(codeword, dtype=np.int32)
        
        # Verify it's valid
        syndrome = (self.H @ codeword) % 2
        if syndrome.sum() > 0:
            print(f"  ⚠ ERROR: Generated codeword has non-zero syndrome! {syndrome.sum()}")
            print(f"     This should never happen with message @ G.T (mod 2)")
        
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
    print(f"LDPC Dataset Generation (pyldpc with proper encoding)")
    print(f"{'='*60}")
    print(f"Codeword length (n):     {n}")
    print(f"Code rate (R):           {code_rate}")
    print(f"Message length (k):      {k}")
    print(f"Parity length (m):       {m}")
    print(f"Number of samples:       {num_samples}")
    
    generator = LDPCGenerator(n=n, code_rate=code_rate, seed=seed)
    
    print(f"\nGenerating {num_samples} codewords...")
    data = generator.generate_dataset(num_samples)
    
    # Save to file WITH H matrix and G matrix
    save_dict = {
        'codewords': data,
        'H_matrix': generator.H,  # Save the H matrix
        'G_matrix': generator.G,  # Save the G matrix too!
        'n': n,
        'k': generator.k,
        'm': generator.m,
        'code_rate': code_rate,
        'seed': seed
    }
    torch.save(save_dict, save_path)
    print(f"\n✓ Dataset saved to: {save_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min value: {data.min()}, Max value: {data.max()}")
    print(f"  H matrix shape: {generator.H.shape}")
    print(f"  G matrix shape: {generator.G.shape}")
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
    print("LDPC Dataset Generator - pyldpc with Proper Encoding")
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
    print(f"\nFirst 3 codewords (first 10 bits):")
    for i in range(min(3, len(data))):
        print(f"  Codeword {i+1}: {data[i][:10].numpy()}")
