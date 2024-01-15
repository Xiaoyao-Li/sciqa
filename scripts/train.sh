#!/bin/bash

EXP_NAME=$1
MODEL=$2
TASK=$3

python train.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            model=${MODEL} \
            task=${TASK} \
            model.enable_image=true \
            model.enable_hint=true \
            model.disable_pretrain_image=true \
            model.disable_pretrain_text=true \
            model.freeze_fasterRCNN=false \
            model.global_pool=cls \
            model.classifier_dropout=0.5