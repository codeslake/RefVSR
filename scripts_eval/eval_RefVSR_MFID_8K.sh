#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=2 python -B run.py \
    --mode RefVSR_MFID_8K \
    --config config_RefVSR_MFID_8K \
    --data RealMCVSR \
    --eval_mode qual \
    --vid_name 0024 0074 0121 \
    --ckpt_abs_name ckpt/RefVSR_MFID_8K.pytorch
    #--ckpt_sc
