# Pytorch implementation of ACLS
This is the implementation of the paper "ACLS:Adaptive and Conditional Label Smoothing for Network Calibration".
For detailed information, please check out our paper [[arXiv](https://arxiv.org/abs/2308.11911)].


## Requirements
This repository has been tested with the following libraries:
* Python = 3.6
* PyTorch = 1.8.0


## Getting started

### Datasets
The structure should be organized as follows:

#### Tiny-ImageNet
```bash
└── /dataset/tiny-imagenet-200
    ├── test
    ├── train
    ├── val
    │   ├── images
    │   └── val_annotations.txt
    ├── wninds.txt
    └── words.txt
```

#### ImageNet
```bash
├── /dataset/ILSVRC2012
└── data
    ├── train
    ├── val
    └── ILSVRC2012_devkit_t12
```


### Training


### Testing


## References
Our work is mainly built on [FLSD](https://github.com/torrvision/focal_calibration), [MbLS](https://github.com/by-liu/MbLS), and [CALS](https://github.com/by-liu/CALS). Thanks to the authors!
