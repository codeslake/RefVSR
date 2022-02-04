# Reference-based Video Super-Resolution with Propagative Temporal Fusion Using Multi-Camera Video Triplets
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-blue.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **[Reference-based Video Super-Resolution with Propagative Temporal Fusion Using Multi-Camera Video Triplets (TODO)]()**<br>
> Junyong Lee, Myeonghee Lee, Sunghyun Cho, Seungyong Lee, ArXiv 2022

<p align="left">
  <a href="https://codeslake.github.io/publications/#IFAN">
    <img width=85% src="./assets/teaser.jpg"/>
  </a><br>
</p>

## About the Research
<details>
    <summary><i>Click here</i></summary>
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

1. **Environment setup**
	```bash
	$ git clone https://github.com/codeslake/RefVSR.git
	$ cd RefVSR

	$ conda create -y name RefVSR pythron 3.8 && conda activate RefVSR

    ## Install pytorch (change the version for cudatoolkit accordingly.)
	$ conda installpytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

    ## Install requirements (select one from below.)
    # for CUDA10.2
	$ sh install/install_cudnn102.sh
    # for CUDA11.1
	$ sh install/install_cudnn111.sh
    # for CUDA11.3 (For running small models with pytorch AMP)
	$ sh install/install_cudnn113.sh
	```

	> **Note:**
	>
	> * For using PyTorch AMP, it is recommended to use PyTorch >= 1.10.0 with CUDA11.3. PyTorch < 1.10 is known to have [a problem in running amp with `torch.nn.functional.grid_sample()`](https://github.com/pytorch/pytorch/issues/42218) needed for warping frames.

2. **Datasets**
    * Download and unzip [the RealMCVSR dataset]() under `[DATASET_ROOT]`:

        ```
        ├── [DATASET_ROOT]
        │   ├── RealMCVSR
        │   │   ├── train                       # training set
        │   │   │   ├── HR                      # videos in original resolution 
        │   │   │   │   ├── T                   # telephoto videos
        │   │   │   │   │   ├── 0002            # video clip
        │   │   │   │   │   │   ├── 0000.png    # frames
        │   │   │   │   │   │   ├── ...         
        │   │   │   │   │   ├── ....            
        │   │   │   │   ├── UW                  # ultra-wide-angle videos
        │   │   │   │   ├── W                   # wide-angle videos
        │   │   │   ├── LRx2                    # 2x down sampled videos
        │   │   │   ├── LRx4                    # 4x downsampled videos
        │   │   ├── test                        # testing set
        │   │   │   ├── ...
        │   │   ├── valid                       # validation set
        │   │   │   ├── ...
        ```

        > **Note:**
        >
        > * `[DATASET_ROOT]` is currently set to `/data1/junyonglee`, which can be modified by [`config.data_offset`](https://github.com/codeslake/RefVSR/blob/main/configs/config.py#L56) in `./configs/config.py`.
3. **Pre-trained models**

## Testing models of ArXiv2022
```shell
## Real-world 4x super-resolution (HD to 8K)
# Ours in Fig. 8 of the main paper (the model trained with pre-training and adaptaion stages)
sh ./scripts_eval/eval_RefVSR_MFID_8K.sh

# A smaller model (not shown in the paper, requires CUDA 11.3 and PyTorch > 1.10.x)
sh ./scripts_eval/eval_amp_RefVSR_small_MFID_8K.sh
```

> **Note:**
>
> * For the first model, GPU with 48GB memory is required. We use Nvidia Quadro 8000 in practice.
> * For the second smaller model, GPU with 24GB memory is required. We use Nvidia GeForce RTX 3090 in practice.


```shell
## Table 2 in the main paper (models trained with the pre-training stage)
# Ours
sh ./scripts_eval/eval_RefVSR_MFID.sh

# Ours-l1
sh ./scripts_eval/eval_RefVSR_L1.sh

# Ours-small-l1
sh ./scripts_eval/eval_amp_RefVSR_small_L1.sh

# Ours-IR-l1
sh ./scripts_eval/eval_RefVSR_IR_L1.sh

```

> **Note:**
>
> * For all models, GPU with 24GB memory is required. We use Nvidia GeForce RTX 3090 in practice.

## Wiki
* [Logging](https://github.com/codeslake/RefVSR/wiki/Log-Details)
* [Training and testing details](https://github.com/codeslake/RefVSR/wiki/Training-&-Testing-Details)

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
| [Checkpoint Files (todo)]() |
| [The RealMCVSR dataset (todo)]() |

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

### Useful Links
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)
