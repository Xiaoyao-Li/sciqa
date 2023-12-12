# sciqa
Project repo of class machine.learning.2023fall, ScienceQA

## Dependencies
```bash
conda create -n sciqa python==3.8.10
conda activate sciqa
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Data Preparation
Download ScienceQA datset by following the [instruction](https://scienceqa.github.io/#dataset). The change your data path in the config.

## Training
### Baseline DFAF
```bash
bash scripts/train.sh exp_dfaf dfaf scienceqa  
```

