#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_MFID \
    --config config_RefVSR_MFID \
    --data RealMCVSR \
    --eval_mode quan_qual \
    --ckpt_abs_name ckpt/RefVSR_MFID.pytorch \
    --is_quan
