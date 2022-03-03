#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9002 run.py \
                        --is_train \
                        --mode RefVSR_L1 \
                        --config config_RefVSR_L1 \
                        --data RealMCVSR \
                        -b 2 \
                        -th 4 \
                        -dl \
                        -dist \
                        -ss \
                        --is_crop_valid \
