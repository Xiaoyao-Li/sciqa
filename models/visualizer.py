import copy
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from omegaconf import DictConfig

from utils.misc import random_str
from utils.registry import Registry
from tqdm import tqdm


VISUALIZER = Registry('Visualizer')

def create_visualizer(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)