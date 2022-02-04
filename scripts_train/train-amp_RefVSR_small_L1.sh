#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=9000 run.py \
                        --is_train \
                        --mode amp_RefVSR_small_L1 \
                        --config config_RefVSR_small_L1 \
                        --network RefVSR \
                        --trainer trainer \
                        --data RealMCVSR \
                        -b 2 \
                        -th 4 \
                        -r 73 \
                        -ss \
                        -dist \
                        -proc 4
