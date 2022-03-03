#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_MFID \
    --config config_RefVSR_MFID \
    --data RealMCVSR \
    --ckpt_abs_name ckpt/RefVSR_MFID.pytorch \
    --data_offset /data1/junyonglee \
    --output_offset ./result
