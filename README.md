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
Download ScienceQA datset by following the [instruction](https://scienceqa.github.io/#dataset). Then change your data path in the config.

## Checkpoint Preparation
Download the checkpoint of the our TFuse model trained on ScienceQA by following the [tsinghua cloud](). Then unzip and organize the checkpoint as follows:
```
├── outputs
│   ├── 2023-12-29_21-28-40_tfuse_scratchrcnn_fullcontext_clspool_do0.5_32bs_1gpu
│   ├── 2024-01-15_11-01-47_tfuse_scratchrcnn_onlyhint_clspool_do0.5_32bs_1gpu
│   ├── 2024-01-15_11-05-36_tfuse_scratchrcnn_onlyimage_clspool_do0.5_32bs_1gpu
│   ├── 2024-01-15_13-27-54_tfuse_scratchrcnn_nocontext_clspool_do0.5_32bs_1gpu
```

## Usage

### Training TFuse Model
```bash
bash scripts/train.sh full_context_32bs_1gpu tfuse scienceqa
```

### Testing Model
```bash
bash scripts/test.sh ${exp_dir}
```

For example, you can directly run the following command to reproduce the result of our TFuse model:
```bash
bash scripts/test.sh outputs/charlie/2023-12-29_21-28-40_tfuse_scratchrcnn_fullcontext_clspool_do0.5_32bs_1gpu
python postprocess/eval_metrics.py --result_file outputs/charlie/2023-12-29_21-28-40_tfuse_scratchrcnn_fullcontext_clspool_do0.5_32bs_1gpu/results/results.json
```