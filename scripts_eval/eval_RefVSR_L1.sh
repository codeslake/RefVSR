#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_L1 \
    --config config_RefVSR_L1 \
    --data RealMCVSR \
    --eval_mode quan_qual \
    --ckpt_abs_name ckpt/RefVSR_L1.pytorch \
    --is_quan \
    #--ckpt_sc
