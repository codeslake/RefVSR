# RefVSR: Reference-based Video Super-Resolution
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-blue.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **[Reference-based Video Super-Resolution Using Multi-Camera Video Triplets (TODO)]()**<br>
> Junyong Lee, Myeonghee Lee, Sunghyun Cho, Seungyong Lee, ArXiv 2022

<p align="left">
<a href="https://codeslake.github.io/publications/#RefVSR">
    <!--<img width=100% src="https://i.imgur.com/DryOd2I.gif"/>-->
    <img width=100% src="https://i.imgur.com/eMz1ZCg.gif"/>
</a><br>
</p>

## About the Research
<details>
    <summary><i>Click here</i></summary>
    <br>
    <p align="left">
    <a href="https://codeslake.github.io/publications/#RefVSR">
        <img width=100% src="https://i.imgur.com/KHwjpaB.png"/>
    </a><br>
    </p>
    <h3> Abstract </h3>
    <p>
       We propose the first reference-based video super-resolution (RefVSR) approach that utilizes reference videos for high-fidelity results. We focus on RefVSR in a triple-camera setting, where we aim at super-resolving a low-resolution ultra-wide video utilizing wide-angle and telephoto videos. We introduce the first RefVSR network that recurrently aligns and propagates temporal reference features fused with features extracted from low-resolution frames. To facilitate the fusion and propagation of temporal reference features, we propose a propagative temporal fusion module. For learning and evaluation of our network, we present the first RefVSR dataset consisting of triplets of ultra-wide, wide-angle, and telephoto videos concurrently taken from triple cameras of a smartphone. We also propose a two-stage training strategy fully utilizing video triplets in the proposed dataset for real-world 4x video super-resolution. We extensively evaluate our method, and the result shows the state-of-the-art performance in 4x super-resolution. 
    </p>
</details>

## Getting Started
### Prerequisites

*Tested environment*

![Ubuntu](https://img.shields.io/badge/Ubuntu-18.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%20&%201.10.2-green.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20&%2011.1%20&%2011.3-green.svg?style=plastic)

#### 1. Environment setup
```bash
$ git clone https://github.com/codeslake/RefVSR.git
$ cd RefVSR

$ conda create -y name RefVSR python 3.8 && conda activate RefVSR

# Install pytorch
$ conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch

# Install requirements
$ ./install/install_cudnn113.sh
```

> It is recommended to install PyTorch >= 1.10.0 with CUDA11.3 for running small models using Pytorch AMP, because PyTorch < 1.10.0 is known to have [a problem in running amp with `torch.nn.functional.grid_sample()`](https://github.com/pytorch/pytorch/issues/42218) needed for inter-frame alignment.

> For the other models, PyTorch 1.8.0 is verified. To install requirements with PyTorch 1.8.0, run `./install/install_cudnn102.sh` for CUDA10.2 or `./install/install_cudnn111.sh` for CUDA11.1.

#### 2. Datasets
Download and unzip the RealMCVSR dataset from [here](https://www.dropbox.com/sh/jyb5tpvctt97s5j/AABUGdkCPPaFU7k5Weo3ivOVa?dl=0) or [here](https://postechackr-my.sharepoint.com/:f:/g/personal/junyonglee_postech_ac_kr/EuWIvNQZWfZMqsE9b375mJQB9jaKrCP4KvgR3uMvCQIdGw?e=5jFIhs) under `[DATASET_ROOT]`:

```
[DATASET_ROOT]
    └── RealMCVSR
        ├── train                       # a training set
        │   ├── HR                      # videos in original resolution 
        │   │   ├── T                   # telephoto videos
        │   │   │   ├── 0002            # a video clip 
        │   │   │   │   ├── 0000.png    # a video frame
        │   │   │   │   └── ...         
        │   │   │   └── ...            
        │   │   ├── UW                  # ultra-wide-angle videos
        │   │   └── W                   # wide-angle videos
        │   ├── LRx2                    # 2x downsampled videos
        │   └── LRx4                    # 4x downsampled videos
        ├── test                        # a testing set
        └── valid                       # a validation set
```

> `[DATASET_ROOT]` can be modified with [`config.data_offset`](https://github.com/codeslake/RefVSR/blob/main/configs/config.py#L56) in `./configs/config.py`.

#### 3. Pre-trained models
Download [pretrained weights](https://www.dropbox.com/sh/pyirbf2jp0uxoq8/AAA6MEuSJYLcwQLVdx3s12Lea?dl=0) under `./ckpt/`:

```
.
├── ...
├── ./ckpt
│   ├── edvr.pytorch                    # weights of EDVR modules used for training Ours-IR
│   ├── SPyNet.pytorch                  # weights of SpyNet used for inter-frame alignment
│   ├── RefVSR_small_L1.pytorch         # weights of Ours-small-L1
│   ├── RefVSR_small_MFID.pytorch       # weights of Ours-small
│   ├── RefVSR_small_MFID_8K.pytorch    # weights of Ours-small-8K
│   ├── RefVSR_L1.pytorch               # weights of Ours-L1
│   ├── RefVSR_MFID.pytorch             # weights of Ours
│   ├── RefVSR_MFID_8K.pytorch.pytorch  # weights of Ours-8K
│   ├── RefVSR_IR_MFID.pytorch          # weights of Ours-IR
│   └── RefVSR_IR_L1.pytorch            # weights of Ours-IR-L1
└── ...
```

---
***For the testing and training of your own model, it is recommended to go through wiki pages for<br>
[Logging](https://github.com/codeslake/RefVSR/wiki/Log-Details) and [Details of testing and training scripts](https://github.com/codeslake/RefVSR/wiki/Details-of-Testing-&-Training-scripts) before running the scripts.***

## Testing models of ArXiv2022
⚠️ Be sure to modify the script file and indicate proper GPU device by modifying `CUDA_VISIBLE_DEVICES`.

⚠️ Results will be saved in `[LOG_ROOT]/RefVSR_CVPR2022/[mode]/result/quan_qual/[mode]_[epoch]/[data]`.
* `[LOG_ROOT]` can be modified with [`config.log_offset`](https://github.com/codeslake/RefVSR/blob/main/configs/config.py#L71) in `./configs/config.py`.
* `[mode]` is the name of the model assigned with the option `--mode` in an evalution script.

### Real-world 4x video super-resolution (HD to 8K resolution)
```bash
# Evaluating the model 'Ours' (Fig. 8 in the main paper).
$ ./scripts_eval/eval_RefVSR_MFID_8K.sh

# Evaluating the model 'Ours-small'.
$ ./scripts_eval/eval_amp_RefVSR_small_MFID_8K.sh
```
> For the model `Ours`, we use Nvidia Quadro 8000 (48GB) in practice.

> For the model `Ours-small`,
> * We use Nvidia GeForce RTX 3090 (24GB) in practice.
> * It is the model `Ours-small` in Table 2 further trained with the adaptation stage.
> * The model requires PyTorch >= 1.10.0 with CUDA 11.3 for using PyTorch AMP.


### Quantitative evaluation (models trained with the pre-training stage)
```bash
## Table 2 in the main paper
# Ours
$ ./scripts_eval/eval_RefVSR_MFID.sh

# Ours-l1
$ ./scripts_eval/eval_RefVSR_L1.sh

# Ours-small
$ ./scripts_eval/eval_amp_RefVSR_small_MFID.sh

# Ours-small-l1
$ ./scripts_eval/eval_amp_RefVSR_small_L1.sh

# Ours-IR
$ ./scripts_eval/eval_RefVSR_IR_MFID.sh

# Ours-IR-l1
$ ./scripts_eval/eval_RefVSR_IR_L1.sh
```
> For all models, we use Nvidia GeForce RTX 3090 (24GB) in practice.

> To obtain quantitative results measured with the varying FoV ranges as shown in Table 3 of the main paper, modify the script and change `--eval_mode quan_qual` to `--eval_mode FOV`.


## Training models with the proposed two-stage training strategy

### The pre-training stage (Sec. 4.1)

```bash
# To train the model 'Ours':
$ ./scripts_train/train_RefVSR_MFID.sh

# To train the model 'Ours-small':
$ ./scripts_train/train_amp_RefVSR_small_MFID.sh
```
> For both models, we use Nvidia GeForce RTX 3090 (24GB) in practice.

> Be sure to modify the script file and set proper GPU devices, number of GPUs, and batch size by modifying `CUDA_VISIBLE_DEVICES`, `--nproc_per_node` and `-b` options, respectively.
> * We use the ***total batch size of 4***, the multiplication of numbers in options `--nproc_per_node` and `-b`.


### The adaptation stage (Sec. 4.2)
1. Set the path of the checkpoint of a model trained with the pre-training stage.<br>For the model `Ours-small`, for example,
    ```bash
    $ vim ./scripts_train/train_amp_RefVSR_small_MFID_8K.sh
    ```
    ```bash
    #!/bin/bash

    py3clean ./
    CUDA_VISIBLE_DEVICES=0,1 ...
        ...
        -ra [LOG_ROOT]/RefVSR_CVPR2022/amp_RefVSR_small_MFID/checkpoint/train/epoch/ckpt/amp_RefVSR_small_MFID_00xxx.pytorch
        ...

    ```
    > Checkpoint path is `[LOG_ROOT]/RefVSR_CVPR2022/[mode]/checkpoint/train/epoch/[mode]_00xxx.pytorch`.
    > * PSNR is recorded in `[LOG_ROOT]/RefVSR_CVPR2022/[mode]/checkpoint/train/epoch/checkpoint.txt`.
    > * `[LOG_ROOT]` can be modified with [`config.log_offset`](https://github.com/codeslake/RefVSR/blob/main/configs/config.py#L71) in `./configs/config.py`.
    > * `[mode]` is the name of the model assigned with `--mode` in the script used for the pre-training stage.

2. Start the adaptation stage.

    ```shell
    # Training the model 'Ours'.
    $ ./scripts_train/train_RefVSR_MFID_8K.sh

    # Training the model 'Ours-small'.
    $ ./scripts_train/train_amp_RefVSR_small_MFID_8K.sh
    ```

    > For the model `Ours`, we use Nvidia Quadro 8000 (48GB) in practice.

    > For the model `Ours-small`, we use Nvidia GeForce RTX 3090 (24GB) in practice.

    > Be sure to modify the script file to set proper GPU devices, number of GPUs, and batch size by modifying `CUDA_VISIBLE_DEVICES`, `--nproc_per_node` and `-b` options, respectively.
    > * We use the ***total batch size of 2***, the multiplication of numbers in options `--nproc_per_node` and `-b`.

## Training models with L1 loss
```bash
# To train the model 'Ours-l1':
$ ./scripts_train/train_RefVSR_L1.sh

# To train the model 'Ours-small-l1':
$ ./scripts_train/train_amp_RefVSR_small_L1.sh

# To train the model 'Ours-IR-l1':
$ ./scripts_train/train_amp_RefVSR_small_L1.sh
```
> For all models, we use Nvidia GeForce RTX 3090 (24GB) in practice.

> Be sure to modify the script file and set proper GPU devices, number of GPUs, and batch size by modifying `CUDA_VISIBLE_DEVICES`, `--nproc_per_node` and `-b` options, respectively.
> * We use the ***total batch size of 8***, the multiplication of numbers in options `--nproc_per_node` and `-b`.


## Wiki
* [Logging](https://github.com/codeslake/RefVSR/wiki/Log-Details)
* [Details of testing and training scripts](https://github.com/codeslake/RefVSR/wiki/Details-of-Testing-&-Training-scripts)

## Citation
If you find this code useful, please consider citing:
```
@artical{Lee_2022_RefVSR,
    author = {Lee, Junyong and Lee, Myeonghee and Cho, Sunghyun and Lee, Seungyong},
    title = {Reference-based Video Super-Resolution with Propagative Temporal Fusion Using Multi-Camera Video Triplets},
    journal = {ArXiv},
    year = {2022}
}
```

## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources
All material related to our paper is available with the following links:

| Link |
| :-------------- |
| [The main paper (todo)]() |
| [Supplementary Files (todo)]() |
| [Checkpoints](https://www.dropbox.com/sh/pyirbf2jp0uxoq8/AAA6MEuSJYLcwQLVdx3s12Lea?dl=0) |
| The RealMCVSR dataset ([option 1](https://www.dropbox.com/sh/jyb5tpvctt97s5j/AABUGdkCPPaFU7k5Weo3ivOVa?dl=0), [option 2](https://postechackr-my.sharepoint.com/:f:/g/personal/junyonglee_postech_ac_kr/EuWIvNQZWfZMqsE9b375mJQB9jaKrCP4KvgR3uMvCQIdGw?e=5jFIhs)) |

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## Acknowledgement
We thank the authors of [BasicVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR) and [DCSR](https://github.com/Tengfei-Wang/DCSR) for sharing their code.

