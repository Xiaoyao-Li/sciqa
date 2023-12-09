import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig

from utils.registry import Registry

EVALUATOR = Registry('Evaluator')

