"""
Simple MLP-based score/energy model for discrete diffusion.
Parameterizes the score function s(x_t, sigma).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                              -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Positional encoding of same shape
        """
        return self.pe[:, :x.size(1), :]


class TimeEmbedding(nn.Module):
    """Embedding for continuous time/sigma parameter."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, sigma):
        """
        Args:
            sigma: Noise level, shape (batch,) or (batch, 1)
        
        Returns:
            Time embedding, shape (batch, dim)
        """
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)  # (batch, 1)
        
        # Simple sinusoidal encoding
        t_embed = torch.sin(self.linear1(sigma))
        t_embed = self.linear2(t_embed)
        return t_embed


class ScoreModel(nn.Module):
    """
    MLP-based score model for discrete diffusion.
    
    Input: x_t (discrete sequence tokens)
    Output: Score function s(x_t, sigma) = log p(x_0 | x_t)
    
    For vocabulary_size=2 (binary), outputs 2 logits per position.
    """
    
    def __init__(self, 
                 vocab_size=2,
                 seq_len=128,
                 embedding_dim=64,
                 hidden_dim=256,
                 num_layers=3,
                 use_positional_encoding=True,
                 dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary (e.g., 2 for binary)
            seq_len: Sequence length
            embedding_dim: Embedding dimension for tokens
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            use_positional_encoding: Whether to add positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embedding_dim, max_len=seq_len)
        
        # Time embedding
        self.time_embed = TimeEmbedding(embedding_dim)
        
        # MLP layers
        layers = []
        
        # First layer: embedding_dim * 2 (tokens + time) -> hidden_dim
        layers.append(nn.Linear(embedding_dim * 2, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer: hidden_dim -> vocab_size (logits)
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, sigma):
        """
        Compute score function for discrete diffusion.
        
        Args:
            x: Discrete token sequence, shape (batch_size, seq_len)
                Values in [0, vocab_size)
            sigma: Noise level, shape (batch_size,)
        
        Returns:
            Score (log-probabilities), shape (batch_size, seq_len, vocab_size)
        """
        batch_size = x.size(0)
        
        # Embed tokens
        x_embed = self.token_embed(x.long())  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x_embed = x_embed + self.pos_encoding(x_embed)
        
        # Embed time/sigma
        sigma_embed = self.time_embed(sigma)  # (batch, embedding_dim)
        
        # Broadcast sigma embedding to all positions
        sigma_embed = sigma_embed.unsqueeze(1).expand(batch_size, self.seq_len, -1)
        # (batch, seq_len, embedding_dim)
        
        # Concatenate embeddings
        combined = torch.cat([x_embed, sigma_embed], dim=-1)  # (batch, seq_len, 2*embedding_dim)
        
        # Reshape for MLP: (batch*seq_len, 2*embedding_dim)
        combined = combined.reshape(batch_size * self.seq_len, -1)
        
        # Pass through MLP
        output = self.mlp(combined)  # (batch*seq_len, vocab_size)
        
        # Reshape back: (batch, seq_len, vocab_size)
        output = output.reshape(batch_size, self.seq_len, self.vocab_size)
        
        return output


class TransformerScoreModel(nn.Module):
    """
    Transformer-based score model for discrete diffusion.
    More expressive than MLP but requires more compute.
    """
    
    def __init__(self,
                 vocab_size=2,
                 seq_len=128,
                 embedding_dim=64,
                 num_heads=4,
                 num_layers=2,
                 ff_dim=256,
                 dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            seq_len: Sequence length
            embedding_dim: Embedding and model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=seq_len)
        
        # Time embedding
        self.time_embed = TimeEmbedding(embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x, sigma):
        """
        Args:
            x: Discrete token sequence, shape (batch_size, seq_len)
            sigma: Noise level, shape (batch_size,)
        
        Returns:
            Score, shape (batch_size, seq_len, vocab_size)
        """
        batch_size = x.size(0)
        
        # Embed tokens
        x_embed = self.token_embed(x.long())  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding(x_embed)
        
        # Embed time
        sigma_embed = self.time_embed(sigma)  # (batch, embedding_dim)
        sigma_embed = sigma_embed.unsqueeze(1).expand(batch_size, self.seq_len, -1)
        
        # Add time embedding to input
        x_embed = x_embed + sigma_embed
        
        # Pass through transformer
        output = self.transformer(x_embed)  # (batch, seq_len, embedding_dim)
        
        # Project to vocabulary size
        output = self.output_proj(output)  # (batch, seq_len, vocab_size)
        
        return output


def create_score_model(model_type='mlp',
                       vocab_size=2,
                       seq_len=128,
                       **kwargs):
    """
    Factory function to create score models.
    
    Args:
        model_type: 'mlp' or 'transformer'
        vocab_size: Vocabulary size
        seq_len: Sequence length
        **kwargs: Additional arguments passed to model
    
    Returns:
        Score model instance
    """
    if model_type == 'mlp':
        return ScoreModel(vocab_size=vocab_size, seq_len=seq_len, **kwargs)
    elif model_type == 'transformer':
        return TransformerScoreModel(vocab_size=vocab_size, seq_len=seq_len, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
