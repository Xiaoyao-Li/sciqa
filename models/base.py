from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry

MODEL = Registry('Model')

def create_model(cfg: DictConfig, slurm: bool) -> nn.Module:
    """ Create a eps model for predicting epsilon

    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        Model
    """
    return MODEL.get(cfg.name)(cfg, slurm)
