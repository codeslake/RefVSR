#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=9001 run.py \
                        --is_train \
                        --mode amp_RefVSR_small_MFID_8K \
                        --config config_RefVSR_small_MFID_8K \
                        --network RefVSR \
                        --trainer trainer \
                        --data RealMCVSR \
                        -b 1 \
                        -th 4 \
                        -dl \
                        -dist \
                        -proc 4
                        #-ra ckpt/RefVSR_MFID.pytorch \
