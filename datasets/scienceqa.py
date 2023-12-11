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
class ScienceQA(Dataset):
    """ Dataset for ScienceQA.
    """

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, **kwargs: Dict) -> None:
        super(ScienceQA, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase in ['train', 'val', 'test', 'trainval', 
                          'minitrain', 'minval', 'minitest']:
            self.splits = [self.phase]
        elif self.phase == 'all':
            self.splits = ['train', 'val', 'test']
        else:
            raise Exception('Unsupported phase.')
        self.device = cfg.device
        self.max_choices = cfg.max_choices
        self.answerable_only = cfg.answerable_only
        self.valid_choices = None
        if self.answerable_only:
            choice_vocab = json.load(open(cfg.vocab_path, 'r'))['choice']
            self.valid_choices = list(choice_vocab.keys())[:self.max_choices]

        ## resource folders
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir
        with open(os.path.join(self.data_dir, 'problems.json'), 'r') as f:
            self.metadatas = json.load(f)
        with open(os.path.join(self.data_dir, 'pid_splits.json'), 'r') as f:
            self.pid_splits = json.load(f)
            self.pid_splits = {k: v for k, v in self.pid_splits.items() if k in self.splits}

        self._pre_load_data()

    def _pre_load_data(self, ) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.indices = []
        for split in self.splits:
            for pid in tqdm(self.pid_splits[split], desc=f'Pre-load {split} split'):
                self.indices.append(pid)
        if self.answerable_only:
            indices = []
            for pid in self.indices:
                metadata = self.metadatas[pid]
                if metadata['choices'][metadata['answer']] in self.valid_choices:
                    indices.append(pid)
            self.indices = indices

        print('Finishing pre-load in ScienceQA.')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """ Get a sample from the dataset
        """
        pid = self.indices[index]
        metadata = self.metadatas[pid]
        data = {}
        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    pass