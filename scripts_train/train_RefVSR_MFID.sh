#!bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=4,5,6,7 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9001 run.py \
                        --is_train \
                        --mode RefVSR_MFID \
                        --config config_RefVSR_MFID \
                        --network RefVSR \
                        --trainer trainer \
                        --data RealMCVSR \
                        --is_crop_valid \
                        -b 1 \
                        -th 4 \
                        -dl \
                        -ss \
                        -dist \
                        -proc 4
