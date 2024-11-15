# MFDIN
> [Multiframe Joint Enhancement for Early Interlaced Videos](https://ieeexplore.ieee.org/abstract/document/9905474)

1st. of MSU Deinterlacer Benchmark since 10.2021 (Half Rate, MOS scores first!).
> [MSU Deinterlacer Benchmark — selecting the best deinterlacing filter](https://videoprocessing.ai/benchmarks/deinterlacer.html)

## What's New
- Dec. 2024:
- Update all codes.

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)   (PyTorch1.1 is confirmed to be available)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [Charles Shang](https://github.com/CharlesShang)'s [DCNv2](https://github.com/CharlesShang/DCNv2) implementation. Please first compile it.
  ```
  cd ./codes/models/archs/dcn
  python setup.py build develop
  ```
- Python packages: `pip install numpy opencv-python lmdb pyyaml`

## Dataset Preparation
We provide some test sets:  [download here](https://drive.google.com/drive/folders/1RjQQW8wO4FPX__srzMqAAZNzHs8-oL64?usp=sharing)

Synthetic test set:  ```../datasets/yk_test```

Real old videos:  ```../datasets/real-old-videos```

## Get Started
- To train the model
  ```
  python codes/train.py
  ```

- Reproduce results with PSNR/SSIM
  ```
  python codes/test_with_GT.py
  ```

- Reproduce the effect of the real old videos
  ```
  python codes/test_only2p.py
  ```

- The results will be found at：[partially inference results](https://drive.google.com/drive/folders/1n_5mwN3I9Nexqt00qJnxw4pVmuqTD3lW?usp=sharing)
  ```
  cd ./results
  ```
## The pretrained model
  ```
  experiments/pretrained_models/MFDIN_old_2P.pth
  
  ```

## Citation

If you find MFDIN useful in your research, please consider cite:

```bibtex
@article{zhao2022multiframe,
  title={Multiframe joint enhancement for early interlaced videos},
  author={Zhao, Yang and Ma, Yanbo and Chen, Yuan and Jia, Wei and Wang, Ronggang and Liu, Xiaoping},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={6282--6294},
  year={2022},
  publisher={IEEE}
}
