#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=7 python -B run.py \
    --mode BasicVSRx4_L1 \
    --eval_mode 'qual_quan' \
    --ckpt_sc

#--mode VRefSRx4_conf_prop_ML_TFID_8K
#--mode VRefSRx4_conf_prop_ML_refs_SRA_4k \
#--mode VRefSRx4_conf_prop_MLT_TFID_4k \
#--mode VRefSRx4_conf_prop_ML_refs_SRA_4k \
#--mode VRefSRx4_ML_no_ref \
