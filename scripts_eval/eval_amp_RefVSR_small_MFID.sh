#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode amp_RefVSR_MFID_small \
    --config config_RefVSR_small_MFID\
    --data RealMCVSR \
    --eval_mode quan_qual \
    --ckpt_sc
    #--ckpt_abs_name ckpt/RefVSR_MFID.pytorch
