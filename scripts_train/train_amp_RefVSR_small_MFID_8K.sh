#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=9001 run.py \
                        --is_train \
                        --mode amp_RefVSR_small_MFID_8K \
                        --config config_RefVSR_small_MFID_8K \
                        -b 1 \
                        -th 4 \
                        -ra ckpt/RefVSR_small_MFID.pytorch \
                        -ss \
                        -dist \
