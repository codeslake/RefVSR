#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_L1 \
    --config config_RefVSR_L1 \
    --data RealMCVSR \
    --ckpt_abs_name ckpt/RefVSR_L1.pytorch \
    --data_offset /data1/junyonglee \
    --output_offset ./result
