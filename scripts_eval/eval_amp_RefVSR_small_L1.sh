#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode amp_RefVSR_small_L1 \
    --config config_RefVSR_small_L1 \
    --data RealMCVSR \
    --eval_mode quan_qual \
    --is_quan \
    --ckpt_sc
    #--ckpt_epoch 72 \
    #--is_qual \
    #--ckpt_abs_name ckpt/RefVSR_MFID.pytorch
