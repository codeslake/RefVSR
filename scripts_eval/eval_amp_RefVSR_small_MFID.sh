#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=1 python -B run.py \
    --mode amp_RefVSR_small_MFID \
    --config config_RefVSR_small_MFID\
    --data RealMCVSR \
    --eval_mode quan_qual \
    --is_quan \
    --ckpt_sc
    #--ckpt_abs_name ckpt/RefVSR_MFID.pytorch
