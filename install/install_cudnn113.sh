#!/bin/bash
pip install --no-cache -r install/requirements.txt

torch_version=$(python -c "import torch; print(torch.__version__)")
torch_version=( ${torch_version//./ } )
torch_version="${torch_version[0]}.${torch_version[1]}"
if [ $torch_version == '1.10' ] || [ $torch_version == '1.11' ]
then
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
elif [ $torch_version == '1.9' ]
then
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html
elif [ $torch_version == '1.8' ]
then
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.8.0/index.html
fi
