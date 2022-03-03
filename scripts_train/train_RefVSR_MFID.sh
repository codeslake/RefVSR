#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9001 run.py \
                        --is_train \
                        --mode RefVSR_MFID \
                        --config config_RefVSR_MFID \
                        --data RealMCVSR \
                        -b 1 \
                        -th 4 \
                        -dl \
                        -dist \
                        -ss \
                        --is_crop_valid \
