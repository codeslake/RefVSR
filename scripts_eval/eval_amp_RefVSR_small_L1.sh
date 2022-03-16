#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=4 python -B run.py \
    --mode amp_RefVSR_small_L1 \
    --config config_RefVSR_small_L1 \
    --data RealMCVSR \
    --ckpt_abs_name ckpt/RefVSR_small_L1.pytorch \
    --data_offset /data1/junyonglee \
    --output_offset ./result \
