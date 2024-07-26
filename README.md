
<p align="center" width="100%">
<img src="assets/title.png"  width="90%">
</p>


<div align=center>
<a href="" target="_blank">
    <img alt="TokenPacker-v1" src="https://img.shields.io/badge/TokenPaker-v1-BFE57E" height="25" />
</a>
<a href="https://arxiv.org/abs/2407.02392" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2407.02392-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/collections/sunshine-lwt/tokenpacker-66a234618f0d2327e0cf2cb1" target="_blank">
    <img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-HuggingFace-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://zhuanlan.zhihu.com/p/707021763" target="_blank">
    <img alt="ZhiHu" src="https://img.shields.io/badge/Blog-ZhiHu-1E90FF?logo=zhihu&logoColor=02B5FD" height="25" />
</a>   
 </div>


---

## Comparisons with existing methods 💡
<!-- <img src="./assets/compare.png" width="80%"> -->
<p align="center" width="100%">
<img src="./assets/compare.jpg"  width="60%">
</p>

## Updates 📌
- [2024/7/25] We released [checkpoints](https://huggingface.co/collections/sunshine-lwt/tokenpacker-66a234618f0d2327e0cf2cb1), please check them.
- [2024/7/3] We released the [paper](https://arxiv.org/abs/2407.02392) of our TokenPacker on Arxiv.
- [2024/7/3] We released the training and inference codes. 


## What is TokenPacker 👀
TokenPacker is a novel visual projector, which adopts a `coarse-to-fine` scheme
to inject the enriched characteristics to generate the condensed visual tokens. Using TokenPacker, we can compress the
visual tokens by **75%∼89%**, while achieves comparable or even better performance
across diverse benchmarks with significantly higher efficiency.
<img src="./assets/framework2.jpg" width="800px">

#### Comparisons with various projectors 
<img src="./assets/projector_comparsion.jpg" width="800px">


## High-Resolution Image Understanding with TokenPacker 🔬
To support efficient `high-resolution` image understanding, we further develop an effective image
cropping method `TokenPacker-HD`.
<img src="./assets/hd.png" width="800px">


## Install 🛠️
1. Clone this repository and navigate to TokenPacker folder
```
git clone https://github.com/CircleRadon/TokenPacker.git
cd TokenPacker
```
2. Install packages
```
conda create -n tokenpacker python=3.10 -y
conda activate tokenpacker
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Training 🚀

### LLaVA-TokenPacker

#### Dataset
To make a fair comparison, we use the same training data as in [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), i.e., [CC3M-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for stage 1, and  [Mix665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) for stage 2.

#### Training 
- Stage1: Image-Text Alignment Pre-training
```shell
bash scripts/v1_5/pretrain.sh
```
- Stage2: Visual Instruction Tuning
```shell
bash scripts/v1_5/finetune.sh
```
Note: Using `--scale_factor` to control compression ratio, support [2,3,4]

### LLaVA-TokenPacker-HD

#### Dataset
To obtain the competitive high-resolution performance, we use 2.7M data as organized by [Mini-Gemini](https://github.com/dvlab-research/MGM#Dataset), i.e., 1.2M for stage 1 and 1.5M for stage 2.

#### Training 
- Stage1: Image-Text Alignment Pre-training
```shell
bash scripts/v1_5/pretrain_hd.sh
```
- Stage2: Visual Instruction Tuning
```shell
bash scripts/v1_5/finetune_hd.sh
```

Note: 
1. Using `--scale_factor` to control compression ratio, support [2,3,4]. 
2. Using `--patch_num` to control max patch dividing number, support [9,16,25].


## Experiments

<img src="./assets/ex1.png" width="800px">

<img src="./assets/high-reso.jpg" width="800px">


## Model Zoo

| Model              |  Max Res.   |  Compre. Ratio  |  Token Num.  |  Max Patch Num.  |                                           Training Data                                            | Download                                                                              |
|--------------------|:-----------:|:---------------:|:------------:|:----------------:|:--------------------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------|
| TokenPacker-7b     |   336x336   |       1/4       |     144      |        -         |                                             558K+665K                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-7b-144token/tree/main)  |
| TokenPacker-7b     |   336x336   |       1/4       |     144      |        -         |                                             558K+665K                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-13b-144token/tree/main) |
| TokenPacker-HD-7b  |  1088x1088  |       1/4       |     ~954     |        9         |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-HD-7b-9patch-144token/tree/main) |
| TokenPacker-HD-13b |  1088x1088  |       1/4       |     ~954     |        9         |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-HD-13b-9patch-144token/tree/main) |
| TokenPacker-HD-13b |  1344x1344  |       1/4       |    ~1393     |        16        |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-HD-13b-16patch-144token/tree/main) |
| TokenPacker-HD-13b |  1344x1344  |       1/9       |     ~619     |        16        |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-HD-13b-16patch-64token/tree/main)                                                                       |
| TokenPacker-HD-13b |  1344x1344  |      1/16       |     ~347     |        16        |                                             1.2M+1.5M                                              |  [checkpoints](https://huggingface.co/sunshine-lwt/TokenPacker-HD-13b-16patch-36token/tree/main)                                                                      |

Note: 
- The `token number` of TokenPacker-HD is the `average` statistically across all training and test data.
- The training data of `558K+665K` follows LLaVA-1.5, the one of `1.2M+1.5M` follows Mini-Gemini.
- All LLMs use Vicuna-7b/13b  as based LLM.


## Visualization
We provide some visual examples.

<img src="./assets/vis-1.jpg" width="800px">


High-resolution image understanding.
<img src="./assets/vis-2.jpg" width="800px">


## TODO List 📝
- [x] Release the training and inference codes.
- [x] Release all checkpoints.


## Acknowledgement 💌
- [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [Mini-Gemini](https://github.com/dvlab-research/MGM): the organized data we used for training high-resolution method.
  


## BibTeX 🖊️
```
@misc{TokenPacker,
  title={TokenPacker: Efficient Visual Projector for Multimodal LLM},
  author={Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu and Lei Zhang},
  year={2024},
  eprint={2407.02392},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
