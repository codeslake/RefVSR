#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B run.py \
    --mode RefVSR_MFID_8K \
    --config config_RefVSR_MFID_8K \
    --data RealMCVSR \
    --ckpt_abs_name ckpt/RefVSR_MFID_8K.pytorch \
    --data_offset /data1/junyonglee \
    --output_offset ./result
