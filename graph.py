"""
Minimal discrete diffusion graph for binary sequences.
Extracted and simplified from: models/SEDD/graph_lib.py
"""

import abc
import torch
import torch.nn.functional as F


def sample_categorical(categorical_probs, method="hard"):
    """Sample from categorical distribution using Gumbel-max trick."""
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


def unsqueeze_as(x, y, back=True):
    """Unsqueeze x to match y's number of dimensions."""
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):
    """
    Abstract base class for discrete diffusion graphs.
    Defines the transition matrices and score functions for discrete diffusion models.
    """

    @property
    @abc.abstractmethod
    def dim(self):
        """Vocabulary/state space dimension."""
        pass

    @property
    def absorb(self):
        """Whether the last state is absorbing (typically used for masking)."""
        return False

    @abc.abstractmethod
    def rate(self, i):
        """
        Compute the i-th column of the rate matrix Q.
        Used for forward diffusion: p(X_t | X_0 = i)
        
        Args:
            i: Current state tensor of shape (...,)
        Returns:
            Rate matrix of shape (..., dim)
        """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Compute the i-th row of the rate matrix Q.
        Used for reverse diffusion.
        
        Args:
            i: Current state tensor
        Returns:
            Transposed rate matrix
        """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Compute transition matrix exp(sigma * Q).
        
        Args:
            i: Current state
            sigma: Diffusion time/noise level
        Returns:
            Transition probabilities of shape (..., dim)
        """
        pass

    def sample_transition(self, i, sigma):
        """Sample next state from transition distribution."""
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """
        Construct reverse rate: score * transp_rate.
        Used in reverse diffusion with score function.
        """
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Transform score for staggered diffusion.
        Implements: p_{sigma-dsigma}(z) / p_sigma(x)
        """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """Sample from the limiting (stationary) distribution."""
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Compute score-based entropy loss for training.
        
        Args:
            score: Model output (log probabilities), shape (..., dim)
            sigma: Noise level
            x: Noisy state (current)
            x0: Clean state (original)
        Returns:
            Loss per sample
        """
        pass


class UniformGraph(Graph):
    """
    Uniform transition graph: everything can transition to everything.
    Probability of transition normalized by dimension to avoid blowup.
    
    Best for binary sequences (dim=2).
    """

    def __init__(self, dim):
        """
        Args:
            dim: Vocabulary size (e.g., 2 for binary)
        """
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def rate(self, i):
        """Rate: each state transitions out uniformly."""
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        """For uniform graph, transp_rate equals rate."""
        return self.rate(i)

    def transition(self, i, sigma):
        """
        Transition probabilities: exp(sigma * Q)
        For uniform: stay with prob exp(-sigma), else go to uniform.
        """
        # Reshape sigma from (batch,) to (batch, 1, 1) for proper broadcasting
        sigma_shaped = sigma[:, None, None] if sigma.dim() == 1 else sigma
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma_shaped).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        """For uniform graph, transp_transition equals transition."""
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        """Sample from transition by Bernoulli(move) then uniform."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        """Transform score for staggered updates."""
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        """Sample from uniform stationary distribution."""
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        """
        Compute score entropy loss.
        
        Negative term: weighted difference between mean score and target score
        Positive term: exponential of scores
        Constant: depends on whether state changed or stayed
        """
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)
        # Reshape ratio from (batch,) to (batch, 1) for broadcasting with (batch, seq_len)
        ratio = ratio[:, None]

        # Negative term: score difference
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        
        # Apply ratio based on whether state stayed or changed
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1[:, None] + neg_term
        )

        # Constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        # Positive term: exponential of scores
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        
        return pos_term - neg_term + const


class AbsorbingGraph(Graph):
    """
    Absorbing graph: everything transitions to an absorbing mask state (dim-1).
    Useful for conditional generation with masking.
    """

    def __init__(self, dim):
        """
        Args:
            dim: Actual vocabulary size (mask state becomes dim+1-1=dim)
        """
        self._dim = dim

    @property
    def dim(self):
        # Return dimension including mask state
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        """Rate: everything goes to absorbing state."""
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        """Transposed rate for absorbing."""
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        """Transition for absorbing: exp(-sigma) to stay, 1-exp(-sigma) to absorb."""
        pass

    def transp_transition(self, i, sigma):
        """Transition probabilities with absorbing state."""
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        """Sample: stay with prob exp(-sigma), else absorb."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(self, score, dsigma):
        """Transform score for absorbing graph."""
        score = score.clone()
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        """Sample from absorbing state (always the mask)."""
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        """Compute score entropy for absorbing graph."""
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        entropy = torch.zeros(*x.shape, device=x.device)
        
        if rel_ind.any():
            ratio = 1 / esigm1.expand_as(x)[rel_ind]
            other_ind = x0[rel_ind]

            # Negative term
            neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

            # Positive term
            pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

            # Constant
            const = ratio * (ratio.log() - 1)

            entropy[rel_ind] += pos_term - neg_term + const
        
        return entropy
