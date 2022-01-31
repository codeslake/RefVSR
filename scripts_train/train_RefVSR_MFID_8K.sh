#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=6,7 python -B -m torch.distributed.launch --nproc_per_node=2 --master_port=9004 run.py \
                        --is_train \
                        --mode RefVSR_MFID_8K \
                        --config config_RefVSR_MFID_8K \
                        --network RefVSR \
                        --trainer trainer \
                        --data RealMCVSR \
                        -b 1 \
                        -th 4 \
                        -ra ckpt/RefVSR_MFID.pytorch \
            			-ss \
                        -dist \
                        -proc 4
