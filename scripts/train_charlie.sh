#!/bin/bash

GPU=$1
EXP_NAME=$2
MODEL=$3
TASK=$4
POOL=$5
DROPOUT=$6

export CUDA_VISIBLE_DEVICES=${GPU}
CHARLIE=1 python train.py hydra/job_logging=none hydra/hydra_logging=none \
                 exp_name=${EXP_NAME} \
                 model=${MODEL} \
                 task=${TASK} \
                 model.enable_image=true \
                 model.enable_hint=true \
                 model.disable_pretrain_image=true \
                 model.disable_pretrain_text=true \
                 model.freeze_fasterRCNN=false \
                 model.global_pool=${POOL} \
                 model.classifier_dropout=${DROPOUT}
