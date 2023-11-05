<h1 align="center"> Instant Photorealistic Style Transfer:<br /> A Lightweight and Adaptive Approach </h1>
<p align="center">
<a style="color:#000000;" href="https://rongliu-leo.github.io/">Rong Liu<sup>1,2</sup></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://www.linkedin.com/in/enyu-zhao-564566250/">Enyu Zhao<sup>1</sup></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://www.linkedin.com/in/liuzy98/">Zhiyuan Liu<sup>1</sup></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://scholar.google.com/citations?user=JKWxGfsAAAAJ&hl=en">Andrew Feng<sup>2</sup></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://viterbi.usc.edu/directory/faculty/Easley/Scott">Scott John Easley<sup>1</sup></a>
</p>

<p align="center">
<a href="https://www.usc.edu/"><sup>1</sup>University of Southern California</a> 
</p>

<p align="center">
<a href="https://ict.usc.edu/"><sup>2</sup>USC Institute for Creative Technologies</a> 
</p>

<p align="center">
<a href="https://rongliu-leo.github.io/Video-Style-Transfer/">Project Page</a>
<a>  |  </a>
<a href="https://arxiv.org/abs/2309.10011">Paper</a> 
</p>




# About
Instant Photorealistic Style Transfer (IPST) approach is designed to achieve 
instant photorealistic style transfer on super-resolution inputs without the need for pre-training on pair-wise datasets or imposing extra constraints. Our method utilizes a lightweight StyleNet to enable style transfer from a style image to a content image while preserving non-color information.
To further enhance the style transfer process, we introduce an instance-adaptive optimization to prioritize the photorealism of outputs and accelerate the convergence of the style network, leading to a rapid training completion within seconds.
Moreover, IPST is well-suited for multi-frame style transfer tasks, as it retains temporal and multi-view consistency of the multi-frame inputs such as video and Neural Radiance Field (NeRF).




# Quickstart
The quickstart will help you install IPST and be familiar with the transfer commands.

## Installation

### Prerequisites

An NVIDIA video card with installed [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Clone repository
```bash
git clone https://github.com/RongLiu-Leo/IPST.git
cd IPST
```

### Create environment

```bash
conda create --name IPST -y python=3.8
conda activate IPST
pip install --upgrade pip
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Image style transfer
For a content-style image pair, run
```bash
python transfer.py --content-image {content_image_path} --style-image {style_image_path}
```

### Video style transfer
By default, IPST will load and transfer all video frames, which means the batch size is equal to the number of frames. This setting achieves extreme speed but can potentially cause memory problems.
```bash
python transfer.py --content-video {content_image_path} --style-image {style_image_path}
```
Using the ```--frame-by-frame``` option will set the batch size to 1, enabling the processing of longer videos at the expense of speed.
```bash
python transfer.py --content-video {content_video_path} --style-image {style_image_path} --frame-by-frame True
```
If it is still not working, try to split the whole video into smaller videos.

# Citation
```bibtex
@misc{liu2023instant,
    title={Instant Photorealistic Style Transfer: A Lightweight and Adaptive Approach}, 
    author={Rong Liu and Enyu Zhao and Zhiyuan Liu and Andrew Wei-Wen Feng and Scott John Easley},
    year={2023},
    eprint={2309.10011},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```