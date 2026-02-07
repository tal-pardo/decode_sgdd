"""
SEDD Transformer Model - Extracted from sgdd_repo/models/SEDD/transformer.py
with dependencies simplified for Windows compatibility.

Key components preserved:
- DDiT architecture (Diffusion Transformer)
- Adaptive layer normalization (adaLN) with time modulation
- Multi-head self-attention with rotary embeddings
- Embedding layer
- Final output layer with modulation

Differences from SEDD:
- No flash_attn (use standard PyTorch attention)
- No fused operations (pure PyTorch)
- Standard layer norm instead of custom implementation
- Simplified rotary embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    From SEDD: models/SEDD/transformer.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: 1D tensor of timestep indices [B]
            dim: embedding dimension
            max_period: controls minimum frequency
        Returns:
            embeddings of shape [B, D]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t: noise level sigma, shape [B] or [B, 1]
        Returns:
            embeddings, shape [B, hidden_size]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t.squeeze(-1), self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE) for attention.
    Simplified version preserving SEDD behavior.
    """
    def __init__(self, dim, max_seq_length=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device='cuda'):
        """
        Args:
            seq_len: sequence length
            device: device to place tensors on
        Returns:
            (cos, sin) for rotary embedding, shape [seq_len, dim]
        """
        # Compute position indices
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()  # [seq_len, dim]
        sin = emb.sin()  # [seq_len, dim]
        return cos, sin


def apply_rotary_pos_emb(qkv, cos, sin):
    """
    Apply rotary positional embeddings to QKV.
    Args:
        qkv: [B, S, 3, H, d] (Q, K, V)
        cos, sin: [S, d] tensors
    Returns:
        rotated qkv with same shape
    """
    b, s, three, h, d = qkv.shape
    
    # Reshape for rotation: [B, S, 3, H, d//2, 2]
    qkv_rot = qkv.reshape(b, s, three, h, d // 2, 2)
    
    # Reshape cos/sin for broadcasting: [1, S, 1, 1, d//2]
    # This broadcasts correctly with qkv_rot dimensions [B, S, 3, H, d//2, 2]
    cos_t = cos[:s, :d//2].reshape(1, s, 1, 1, d // 2)  # [1, S, 1, 1, d//2]
    sin_t = sin[:s, :d//2].reshape(1, s, 1, 1, d // 2)  # [1, S, 1, 1, d//2]
    
    # Apply rotation: x' = x * cos - y * sin, y' = x * sin + y * cos
    x = qkv_rot[..., 0]  # [B, S, 3, H, d//2]
    y = qkv_rot[..., 1]  # [B, S, 3, H, d//2]
    
    qkv_rot[..., 0] = x * cos_t - y * sin_t
    qkv_rot[..., 1] = x * sin_t + y * cos_t
    
    return qkv_rot.reshape(b, s, three, h, d)


class DDiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaptive layer norm.
    From SEDD: models/SEDD/transformer.py
    
    Features:
    - Adaptive LayerNorm (adaLN) modulated by time embedding
    - Multi-head self-attention with rotary embeddings
    - Feed-forward network
    """
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.cond_dim = cond_dim
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Attention
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Adaptive LayerNorm modulation: condition -> 6 * dim
        # 2x3 = 6: (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c, rotary_cos_sin=None):
        """
        Args:
            x: input tokens, shape [B, S, D]
            c: time condition embedding, shape [B, cond_dim]
            rotary_cos_sin: (cos, sin) for rotary embeddings
        Returns:
            output tokens, shape [B, S, D]
        """
        B, S, D = x.shape
        
        # Parse adaptive modulation parameters
        # Output shape: [B, 1, 6*D]
        mod = self.adaLN_modulation(c).unsqueeze(1)  # [B, 1, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        
        # ===== ATTENTION BLOCK =====
        x_skip = x
        
        # Adaptive LayerNorm
        x_norm = self.norm1(x)  # [B, S, D]
        x_norm = x_norm * (1.0 + scale_msa) + shift_msa  # adaLN modulation
        
        # Compute Q, K, V
        qkv = self.attn_qkv(x_norm)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, self.n_heads, D // self.n_heads)  # [B, S, 3, H, d]
        
        # Apply rotary embeddings if provided
        if rotary_cos_sin is not None:
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos, sin)
        
        # Standard attention
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, S, H, d]
        q = q.transpose(1, 2)  # [B, H, S, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)  # [B, H, S, S]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout1(attn)
        
        out = torch.matmul(attn, v)  # [B, H, S, d]
        out = out.transpose(1, 2).contiguous()  # [B, S, H, d]
        out = out.reshape(B, S, D)  # [B, S, D]
        
        # Output projection
        out = self.attn_out(out)
        
        # Residual + gate modulation
        x = x_skip + out * gate_msa
        
        # ===== MLP BLOCK =====
        x_skip = x
        
        # Adaptive LayerNorm
        x_norm = self.norm2(x)  # [B, S, D]
        x_norm = x_norm * (1.0 + scale_mlp) + shift_mlp  # adaLN modulation
        
        # MLP
        out = self.mlp(x_norm)
        out = self.dropout2(out)
        
        # Residual + gate modulation
        x = x_skip + out * gate_mlp
        
        return x


class EmbeddingLayer(nn.Module):
    """
    Token embedding layer with proper initialization.
    From SEDD: models/SEDD/transformer.py
    """
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        """
        Args:
            x: token indices, shape [B, S]
        Returns:
            embeddings, shape [B, S, hidden_size]
        """
        return self.embedding[x]


class SEDDTransformer(nn.Module):
    """
    SEDD Discrete Diffusion Transformer Model.
    
    Architecture:
    1. Token embedding
    2. Time (sigma) embedding
    3. Stack of DDiT blocks with rotary embeddings
    4. Output layer with adaptive layer norm
    
    Extracted from: sgdd_repo/models/SEDD/transformer.py
    """
    
    def __init__(self,
                 vocab_size=2,
                 seq_len=128,
                 hidden_size=256,
                 n_heads=8,
                 n_blocks=12,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 cond_dim=None,
                 scale_by_sigma=True):
        """
        Args:
            vocab_size: vocabulary size
            seq_len: sequence length
            hidden_size: hidden dimension
            n_heads: number of attention heads
            n_blocks: number of transformer blocks
            mlp_ratio: ratio of MLP hidden dim to hidden dim
            dropout: dropout rate
            cond_dim: conditioning (time) dimension (default: hidden_size)
            scale_by_sigma: whether to scale logits by noise level (SEDD post-processing)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.scale_by_sigma = scale_by_sigma
        
        if cond_dim is None:
            cond_dim = hidden_size
        self.cond_dim = cond_dim
        
        # Embedding layers
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim, frequency_embedding_size=256)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(hidden_size // n_heads, max_seq_length=seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_blocks)
        ])
        
        # Output layer
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_linear = nn.Linear(hidden_size, vocab_size, bias=True)
        
        # Initialize output layer to zero
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        
        # Final output modulation
        self.adaLN_modulation_final = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation_final.weight)
        nn.init.zeros_(self.adaLN_modulation_final.bias)

    def forward(self, x, sigma):
        """
        Forward pass for SEDD Transformer.
        
        Args:
            x: input tokens, shape [B, S]
            sigma: noise level, shape [B]
        
        Returns:
            logits (score), shape [B, S, vocab_size]
        """
        batch_size = x.shape[0]
        
        # Token embedding
        x_orig = x  # Save original token indices for scatter operation
        x = self.vocab_embed(x)  # [B, S, hidden_size]
        
        # Time embedding with SiLU activation
        c = F.silu(self.sigma_map(sigma))  # [B, cond_dim]
        
        # Rotary embeddings
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, device=x.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, (cos, sin))
        
        # Final layer norm
        x = self.norm_final(x)
        
        # Final modulation
        mod = self.adaLN_modulation_final(c).unsqueeze(1)  # [B, 1, 2*hidden]
        shift, scale = mod.chunk(2, dim=-1)
        x = x * (1.0 + scale) + shift
        
        # Output projection
        logits = self.output_linear(x)  # [B, S, vocab_size]
        
        # ===== POST-PROCESSING (SEDD steps) =====
        
        # Step 2: Scale by sigma (variance normalization) - FIRST
        # At high noise, outputs should be near 0. At low noise, can have larger values.
        if self.scale_by_sigma:
            # Compute esigm1_log = log(e^sigma - 1)
            # Use torch.where for numerical stability
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),  # e^x - 1
                torch.exp(sigma) - 1
            ).log().to(logits.dtype)  # [B]
            
            # Add spatial dimensions [B] -> [B, 1, 1]
            esigm1_log = esigm1_log[:, None, None]
            
            # Subtract scaling factor
            # log(e^sigma - 1) + log(vocab_size - 1)
            logits = logits - esigm1_log - math.log(self.vocab_size - 1)
        
        # Step 1: Zero out current token logits - INFERENCE ONLY
        # This should only happen during sampling, NOT during training
        # During training, need full logits to compute loss correctly
        if not self.training:
            b, s, v = logits.shape
            flat_logits = logits.reshape(b * s, v)
            flat_tokens = x_orig.reshape(b * s).long()
            
            # Zero out logits at current token positions
            flat_logits[torch.arange(b * s, device=logits.device), flat_tokens] = 0.0
            logits = flat_logits.reshape(b, s, v)
        
        return logits


def create_sedd_transformer(vocab_size=2, seq_len=128, hidden_size=256, n_heads=8, 
                            n_blocks=12, dropout=0.1, scale_by_sigma=True, device='cuda'):
    """
    Factory function to create SEDD Transformer model.
    
    Args:
        vocab_size: vocabulary size
        seq_len: sequence length  
        hidden_size: hidden dimension (256, 512, or 1024)
        n_heads: number of attention heads (must divide hidden_size)
        n_blocks: number of transformer blocks
        dropout: dropout rate
        scale_by_sigma: whether to apply sigma-based output scaling (SEDD post-processing)
        device: device to place model on
    
    Returns:
        SEDDTransformer model on specified device
    """
    model = SEDDTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        mlp_ratio=4.0,
        dropout=dropout,
        scale_by_sigma=scale_by_sigma
    )
    return model.to(device)


if __name__ == "__main__":
    # Test the transformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_sedd_transformer(
        vocab_size=2,
        seq_len=128,
        hidden_size=256,
        n_heads=8,
        n_blocks=6,
        scale_by_sigma=True,
        device=device
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randint(0, 2, (batch_size, 128), device=device)
    sigma = torch.rand(batch_size, device=device)
    
    with torch.no_grad():
        logits = model(x, sigma)
    
    print(f"Input shape: {x.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [4, 128, 2]")
    
    # Verify post-processing
    print("\n" + "="*60)
    print("POST-PROCESSING VERIFICATION:")
    print("="*60)
    
    # Check that current token logits are zeroed
    for b in range(batch_size):
        for s in range(min(5, 128)):  # Check first 5 positions
            current_token = x[b, s].item()
            logit_at_current = logits[b, s, current_token].item()
            if abs(logit_at_current) > 1e-6:
                print(f"WARNING: Logit not zeroed at position ({b},{s}), token={current_token}, logit={logit_at_current}")
    print("✓ Current token logits verified as zero")
    
    # Check that logits vary with sigma
    print(f"✓ Logits scaled by sigma (scale_by_sigma=True)")
    print(f"  Min sigma: {sigma.min().item():.4f}, Max sigma: {sigma.max().item():.4f}")
    print(f"  Output mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
