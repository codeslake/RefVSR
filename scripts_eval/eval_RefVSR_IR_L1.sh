#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_IR_L1 \
    --config config_RefVSR_IR_L1 \
    --data RealMCVSR \
    --eval_mode FOV \
    --ckpt_abs_name ckpt/RefVSR_IR_L1.pytorch
    # --eval_mode quan_qual \
    #--ckpt_sc
