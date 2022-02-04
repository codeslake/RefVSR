#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=9000 run.py \
                        --is_train \
                        --mode amp_RefVSR_small_L1 \
                        --config config_RefVSR_small_L1 \
                        -b 2 \
                        -th 4 \
                        -dl \
                        -ss \
                        -dist \
