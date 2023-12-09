import time
from typing import Any, Tuple, Dict
import os
import json
import glob
from tqdm import tqdm
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets.misc import collate_fn_general
# from datasets.transforms import make_default_transform
from datasets.base import DATASET


@DATASET.register()
class SienceQA(Dataset):
    """ Dataset for ScienceQA.
    """

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, **kwargs: Dict) -> None:
        super(SienceQA, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'val':
            self.split = self._val_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.device = cfg.device

        ## resource folders
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir
        with open(os.path.join(self.data_dir, 'desc.json'), 'r') as f:
            self.dataset_desc = json.load(f)

        self._pre_load_data()

    def _pre_load_data(self, ) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.indices = []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = {}
        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    pass