#!/bin/bash
pip install --no-cache -r install/requirements.txt
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

#apt install -y gnupg
#apt update
#apt install -y gcc g++ cpp
#pip install --no-cache cupy-cuda111
#pip install -U fvcore

#cd ./models/archs/correlation_package
#rm -rf *_cuda.egg-info build dist __pycache__
#python setup.py install --user
#python setup_over_cudnn102.py install --user
#cd ../../../
