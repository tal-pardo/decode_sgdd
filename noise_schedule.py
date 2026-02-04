"""
Noise schedules for discrete diffusion.
Extracted and simplified from various config files.
"""

import torch
import numpy as np


class NoiseSchedule:
    """Base class for noise schedules."""
    
    def __call__(self, t):
        """
        Get sigma and dsigma/dt at time t.
        
        Args:
            t: Time parameter, typically in [0, 1]
        
        Returns:
            (sigma, dsigma/dt)
        """
        raise NotImplementedError
    
    def sample_t(self, batch_size, device='cuda'):
        """Sample random time steps."""
        raise NotImplementedError


class CosineSchedule(NoiseSchedule):
    """
    Cosine annealing schedule.
    sigma(t) = sigma_max * cos(pi/2 * t)
    """
    
    def __init__(self, sigma_min=0.001, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, t):
        """
        Compute sigma(t) using cosine schedule.
        
        Args:
            t: Time in [0, 1], shape (batch_size,)
        
        Returns:
            sigma: shape (batch_size,)
            dsigma_dt: shape (batch_size,)
        """
        # Ensure t is tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        t = t.float()
        
        # sigma(t) = sigma_max * cos(pi/2 * t) + sigma_min
        sigma = self.sigma_max * torch.cos(torch.pi / 2 * t) + self.sigma_min
        
        # dsigma/dt = -sigma_max * pi/2 * sin(pi/2 * t)
        dsigma_dt = -self.sigma_max * (torch.pi / 2) * torch.sin(torch.pi / 2 * t)
        
        return sigma, dsigma_dt
    
    def sample_t(self, batch_size, device='cuda'):
        """Sample uniform time steps in [0, 1]."""
        return torch.rand(batch_size, device=device)


class LinearSchedule(NoiseSchedule):
    """
    Linear schedule.
    sigma(t) = sigma_min + (sigma_max - sigma_min) * t
    """
    
    def __init__(self, sigma_min=0.001, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, t):
        """
        Compute sigma(t) using linear schedule.
        
        Args:
            t: Time in [0, 1], shape (batch_size,)
        
        Returns:
            sigma: shape (batch_size,)
            dsigma_dt: shape (batch_size,)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        t = t.float()
        
        # sigma(t) = sigma_min + (sigma_max - sigma_min) * t
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        
        # dsigma/dt = sigma_max - sigma_min (constant)
        dsigma_dt = torch.full_like(t, self.sigma_max - self.sigma_min)
        
        return sigma, dsigma_dt
    
    def sample_t(self, batch_size, device='cuda'):
        """Sample uniform time steps in [0, 1]."""
        return torch.rand(batch_size, device=device)


class ExponentialSchedule(NoiseSchedule):
    """
    Exponential schedule.
    sigma(t) = sigma_min * exp(log(sigma_max/sigma_min) * t)
    """
    
    def __init__(self, sigma_min=0.001, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, t):
        """
        Compute sigma(t) using exponential schedule.
        
        Args:
            t: Time in [0, 1], shape (batch_size,)
        
        Returns:
            sigma: shape (batch_size,)
            dsigma_dt: shape (batch_size,)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        t = t.float()
        
        log_sigma_min = torch.log(torch.tensor(self.sigma_min))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max))
        
        log_sigma = log_sigma_min + (log_sigma_max - log_sigma_min) * t
        sigma = torch.exp(log_sigma)
        
        # dsigma/dt = (log(sigma_max) - log(sigma_min)) * sigma
        dsigma_dt = (log_sigma_max - log_sigma_min) * sigma
        
        return sigma, dsigma_dt
    
    def sample_t(self, batch_size, device='cuda'):
        """Sample uniform time steps in [0, 1]."""
        return torch.rand(batch_size, device=device)


def get_schedule(schedule_type='cosine', sigma_min=0.001, sigma_max=1.0):
    """
    Factory function to get noise schedule.
    
    Args:
        schedule_type: 'cosine', 'linear', or 'exponential'
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
    
    Returns:
        NoiseSchedule instance
    """
    if schedule_type == 'cosine':
        return CosineSchedule(sigma_min, sigma_max)
    elif schedule_type == 'linear':
        return LinearSchedule(sigma_min, sigma_max)
    elif schedule_type == 'exponential':
        return ExponentialSchedule(sigma_min, sigma_max)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
