#!/bin/bash

MODEL=tfuse_scratch
TASK=scienceqa
GPU=$1
EXP_NAME=$2
POOL=$3
DROPOUT=$4

export CUDA_VISIBLE_DEVICES=${GPU}
CHARLIE=1 python train.py hydra/job_logging=none hydra/hydra_logging=none \
                 exp_name=${EXP_NAME} \
                 model=${MODEL} \
                 task=${TASK} \
                 model.enable_image=true \
                 model.enable_hint=true \
                 model.global_pool=${POOL} \
                 model.classifier_dropout=${DROPOUT}
