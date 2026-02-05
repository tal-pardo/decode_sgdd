"""
Split Gibbs Sampler for discrete diffusion inference.
Extracted and simplified from: sampling/sgdd.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DiscreteDiffusionSampler:
    """
    Base sampler for discrete diffusion models.
    Implements the reverse diffusion process.
    """
    
    def __init__(self, model, graph, noise_schedule, device='cuda', num_steps=100):
        """
        Args:
            model: Score model (neural network)
            graph: Discrete diffusion graph (UniformGraph, etc.)
            noise_schedule: Noise schedule (CosineSchedule, etc.)
            device: Device to run on
            num_steps: Number of diffusion steps
        """
        self.model = model
        self.graph = graph
        self.noise_schedule = noise_schedule
        self.device = device
        self.num_steps = num_steps
        
        # Create time steps
        self.time_steps = torch.linspace(0, 1 - 1e-5, num_steps)
    
    @torch.no_grad()
    def sample(self, num_samples=1, observation=None, verbose=True):
        """
        Generate samples using reverse diffusion.
        
        Args:
            num_samples: Number of samples to generate
            observation: Optional conditioning observation
            verbose: Whether to show progress bar
        
        Returns:
            Generated samples of shape (num_samples, seq_len)
        """
        raise NotImplementedError


class UnconditionalSampler(DiscreteDiffusionSampler):
    """
    Simple unconditional sampler using score-based reverse diffusion.
    """
    
    @torch.no_grad()
    def sample(self, num_samples=1, verbose=True):
        """
        Generate unconditional samples.
        
        Args:
            num_samples: Number of samples to generate
            verbose: Whether to show progress bar
        
        Returns:
            Samples of shape (num_samples, seq_len)
        """
        # Start from random/uniform distribution
        x_t = self.graph.sample_limit(num_samples, self.model.seq_len).to(self.device)
        
        iterator = tqdm(reversed(self.time_steps)) if verbose else reversed(self.time_steps)
        
        for t_idx, t in enumerate(iterator):
            # Get current and previous sigma
            sigma_t, _ = self.noise_schedule(torch.tensor([t], device=self.device))
            sigma_t = sigma_t.squeeze()
            
            if t_idx < len(self.time_steps) - 1:
                t_prev = self.time_steps[t_idx + 1]
                sigma_prev, _ = self.noise_schedule(torch.tensor([t_prev], device=self.device))
                sigma_prev = sigma_prev.squeeze()
            else:
                sigma_prev = torch.tensor(0.0, device=self.device)
            
            # Get score prediction
            score = self.model(x_t, sigma_t.expand(num_samples))
            
            # Denoising step using score
            # p(x_{t-1} | x_t) proportional to score * transition matrix
            stag_score = self.graph.staggered_score(score, (sigma_t - sigma_prev).unsqueeze(-1))
            
            # Get transition probabilities
            transition_probs = stag_score * self.graph.transp_transition(x_t, sigma_t.expand(num_samples))
            
            # Normalize and sample
            transition_probs = transition_probs / (transition_probs.sum(dim=-1, keepdim=True) + 1e-10)
            x_t = self._sample_categorical(transition_probs)
        
        return x_t
    
    def _sample_categorical(self, probs):
        """Sample from categorical distribution."""
        batch_size, seq_len, vocab_size = probs.shape
        
        # Flatten for sampling
        probs_flat = probs.reshape(-1, vocab_size)
        
        # Gumbel-max trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs_flat) + 1e-10) + 1e-10)
        samples = (probs_flat.log() + gumbel_noise).argmax(dim=-1)
        
        return samples.reshape(batch_size, seq_len)


class SplitGibbsSampler(DiscreteDiffusionSampler):
    """
    Split Gibbs Sampler for conditional discrete diffusion inference.
    Combines score-based reverse diffusion with Metropolis-Hastings conditioning.
    
    References: https://arxiv.org/abs/2405.18782
    """
    
    def __init__(self,
                 model,
                 graph,
                 noise_schedule,
                 forward_op=None,
                 device='cuda',
                 num_steps=100,
                 mh_steps=50,
                 alpha=1.0,
                 max_hamming_dist=1):
        """
        Args:
            model: Score model
            graph: Diffusion graph
            noise_schedule: Noise schedule
            forward_op: Forward operator for conditioning (optional)
            device: Device
            num_steps: Number of diffusion steps
            mh_steps: Number of Metropolis-Hastings steps per diffusion step
            alpha: Scaling factor for likelihood
            max_hamming_dist: Maximum Hamming distance for MH proposals
        """
        super().__init__(model, graph, noise_schedule, device, num_steps)
        self.forward_op = forward_op
        self.mh_steps = mh_steps
        self.alpha = alpha
        self.max_hamming_dist = max_hamming_dist
    
    @torch.no_grad()
    def sample(self, num_samples=1, observation=None, verbose=True):
        """
        Generate samples using Split Gibbs sampler.
        
        Args:
            num_samples: Number of samples
            observation: Optional conditioning (e.g., observed measurements y)
            verbose: Show progress
        
        Returns:
            Samples of shape (num_samples, seq_len)
        """
        if observation is None and self.forward_op is not None:
            raise ValueError("forward_op provided but observation is None")
        
        # Initialize from prior
        x_t = self.graph.sample_limit((num_samples,) + (self.model.seq_len,)).to(self.device)
        
        iterator = tqdm(range(self.num_steps)) if verbose else range(self.num_steps)
        
        for step_idx in iterator:
            t = self.time_steps[step_idx]
            sigma_t, _ = self.noise_schedule(torch.tensor([t], device=self.device))
            sigma_t = sigma_t.squeeze()
            
            # 1. Score-based reverse step
            score = self.model(x_t, sigma_t.expand(num_samples))
            
            # Staggered score
            if step_idx < len(self.time_steps) - 1:
                t_prev = self.time_steps[step_idx + 1]
                sigma_prev, _ = self.noise_schedule(torch.tensor([t_prev], device=self.device))
                sigma_prev = sigma_prev.squeeze()
                dsigma = sigma_t - sigma_prev
            else:
                dsigma = sigma_t
            
            stag_score = self.graph.staggered_score(score, dsigma.unsqueeze(-1))
            
            # Transition
            transition_probs = stag_score * self.graph.transp_transition(x_t, sigma_t.expand(num_samples))
            transition_probs = transition_probs / (transition_probs.sum(dim=-1, keepdim=True) + 1e-10)
            x_t = self._sample_categorical(transition_probs)
            
            # 2. Metropolis-Hastings conditioning step
            if observation is not None and self.forward_op is not None:
                x_t = self._metropolis_hastings(
                    x_t, observation, sigma_t.expand(num_samples), num_samples
                )
        
        return x_t
    
    def _metropolis_hastings(self, x_current, observation, sigma, num_samples):
        """
        Metropolis-Hastings step for likelihood conditioning.
        
        Args:
            x_current: Current state
            observation: Conditioning observation
            sigma: Current noise level
            num_samples: Batch size
        
        Returns:
            Updated samples
        """
        x = x_current.clone()
        seq_len = x.shape[-1]
        vocab_size = self.graph.dim
        
        # Compute current log-likelihood
        current_ll = self.forward_op.log_likelihood(x, observation)
        
        for _ in range(self.mh_steps):
            # Propose: randomly flip up to max_hamming_dist dimensions
            proposal = x.clone()
            
            for _ in range(self.max_hamming_dist):
                # Random dimension and new value
                idx = torch.randint(seq_len, (num_samples,), device=x.device)
                v = torch.randint(vocab_size, (num_samples,), device=x.device)
                
                proposal[torch.arange(num_samples), idx] = v
            
            # Compute proposal log-likelihood
            proposal_ll = self.forward_op.log_likelihood(proposal, observation)
            
            # Hamming distances
            current_hamming = (x != x_current).sum(dim=-1)
            proposal_hamming = (proposal != x_current).sum(dim=-1)
            
            # Log acceptance ratio
            log_ratio = proposal_ll - current_ll
            log_ratio += self._log_prior_ratio(
                sigma, proposal_hamming, current_hamming, vocab_size
            )
            
            # Metropolis-Hastings accept/reject
            rho = torch.clamp(torch.exp(log_ratio), max=1.0)
            accept = torch.rand_like(rho) < rho
            
            # Update
            x = torch.where(accept.unsqueeze(-1), proposal, x)
            current_ll = torch.where(accept, proposal_ll, current_ll)
        
        return x
    
    def _log_prior_ratio(self, sigma, proposal_hamming, current_hamming, vocab_size):
        """
        Compute log p(proposal) / p(current) under diffusion prior.
        
        Args:
            sigma: Current noise level
            proposal_hamming: Hamming distance of proposal from x_0
            current_hamming: Hamming distance of current from x_0
            vocab_size: Vocabulary size
        
        Returns:
            Log prior ratio
        """
        # Simplified: uniform prior over Hamming distances
        # In full version, would compute exact transition probabilities
        alpha = (1 - torch.exp(-sigma)) * (1 - 1/vocab_size)
        
        log_alpha = torch.log(alpha + 1e-10)
        log_1alpha = torch.log(1 - alpha + 1e-10)
        
        seq_len = self.model.seq_len
        
        log_ratio = (proposal_hamming - current_hamming) * log_alpha
        log_ratio += (current_hamming - proposal_hamming) * log_1alpha
        
        return log_ratio
    
    def _sample_categorical(self, probs):
        """Sample from categorical using Gumbel-max."""
        batch_size, seq_len, vocab_size = probs.shape
        probs_flat = probs.reshape(-1, vocab_size)
        
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs_flat) + 1e-10) + 1e-10)
        samples = (probs.log() + gumbel_noise).argmax(dim=-1)
        
        return samples.reshape(batch_size, seq_len)
