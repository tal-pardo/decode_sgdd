"""
Discrete Diffusion Framework - Binary Edition

A minimal, standalone implementation of discrete diffusion models (SGDD/SEDD)
optimized for training on binary datasets with a single GPU/CPU.
"""

__version__ = "0.1.0"

from .graph import Graph, UniformGraph, AbsorbingGraph, sample_categorical
from .noise_schedule import NoiseSchedule, CosineSchedule, LinearSchedule, get_schedule
from .model import ScoreModel, TransformerScoreModel, create_score_model
from .dataset import BinarySequenceDataset, CustomBinaryDataset, ForwardOperator, MaskingOperator, BinaryOperator
from .sampler import DiscreteDiffusionSampler, UnconditionalSampler, SplitGibbsSampler
from .trainer import Trainer

__all__ = [
    'Graph',
    'UniformGraph',
    'AbsorbingGraph',
    'NoiseSchedule',
    'CosineSchedule',
    'LinearSchedule',
    'get_schedule',
    'ScoreModel',
    'TransformerScoreModel',
    'create_score_model',
    'BinarySequenceDataset',
    'CustomBinaryDataset',
    'ForwardOperator',
    'MaskingOperator',
    'BinaryOperator',
    'DiscreteDiffusionSampler',
    'UnconditionalSampler',
    'SplitGibbsSampler',
    'Trainer',
]
