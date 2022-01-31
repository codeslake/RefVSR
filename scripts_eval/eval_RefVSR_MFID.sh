#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=4 python -B run.py \
    --mode RefVSR_MFID \
    --mode config_RefVSR_MFID \
    --data RealMCVSR \
    --eval_mode quan_qual \
    --ckpt_abs_name ckpt/RefVSR_MFID.pytorch
