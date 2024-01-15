#!/bin/bash

EXP_DIR=$1

python test.py hydra/job_logging=none hydra/hydra_logging=none \
            model=tfuse \
            task=scienceqa \
            model.loss_type=TEST \
            model.global_pool=cls \
            exp_dir=${EXP_DIR} \
            model.enable_image=true \
            model.enable_hint=true