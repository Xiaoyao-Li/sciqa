from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry

MODEL = Registry('Model')