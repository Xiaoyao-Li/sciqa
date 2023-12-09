#!/bin/bash

EXP_NAME=$1
MODEL=$2
TASK=$3

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                model=${MODEL} \
                task=${TASK}