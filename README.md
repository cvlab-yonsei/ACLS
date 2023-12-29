# Pytorch implementation of ACLS
This is the implementation of the paper "ACLS:Adaptive and Conditional Label Smoothing for Network Calibration".

For detailed information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/ACLS/)] or the paper [[arXiv](https://arxiv.org/abs/2308.11911)].



## Requirements
This repository has been tested with the following libraries:
* Python = 3.6
* PyTorch = 1.8.0


## Getting started

### Installation
```bash
pip install -e .
```

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
#### Tiny-ImageNet
```bash
sh tools/run_tiny.sh
```

#### ImageNet
```bash
sh tools/run_imagenet.sh
```

### Testing

#### Tiny-ImageNet
```bash
python tools/test.py data=tiny_imagenet hydra.run.dir=your/weight/directory test.checkpoint=your_weight.pth
```

#### ImageNet
```bash
python tools/test.py data=imagenet hydra.run.dir=your/weight/directory test.checkpoint=your_weight.pth
```

## Pretrained models
**NOTE**: we train networks with one and four A5000 GPU for Tiny-ImageNet and ImageNet, respectively.


<details><summary>Tiny-ImageNet</summary>
  

* [ResNet50](https://drive.google.com/file/d/1IabIHSEYtErC02wJ6ScfeKhfbotsi-_q/view?usp=sharing)
```bash
[2023-02-07 18:31:53,040 INFO][tester.py:124 - log_eval_epoch_info] - 
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 10000   | 0.64840 | 0.85960 | 0.64840 |
+---------+---------+---------+---------+
[2023-02-07 18:31:53,040 INFO][tester.py:125 - log_eval_epoch_info] - 
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 10000   | 1.42108 | 0.01050 | 0.01029 | 0.00135 |
+---------+---------+---------+---------+---------+
```

* [ResNet101](https://drive.google.com/file/d/1sxX4GuTSkdzpTeU76Y7pnAOLsnp6_1Fj/view?usp=sharing)

```bash
[2023-01-21 05:55:53,096 INFO][trainer.py:214 - log_eval_epoch_info] - 
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 10000   | 0.65780 | 0.86330 | 0.65780 |
+---------+---------+---------+---------+
[2023-01-21 05:55:53,096 INFO][trainer.py:215 - log_eval_epoch_info] - 
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 10000   | 1.38089 | 0.01107 | 0.01151 | 0.00131 |
+---------+---------+---------+---------+---------+
```

</p>
</details>


<details><summary>ImageNet</summary>

* [ResNet50](https://drive.google.com/file/d/1fXIo2GU-npLgkFRkvTxnECE9_kqiAGCr/view?usp=sharing)
```bash
[2023-02-10 14:01:17,389 INFO][tester.py:149 - log_eval_epoch_info] -
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 40000   | 0.75650 | 0.92363 | 0.75360 |
+---------+---------+---------+---------+
[2023-02-10 14:01:17,389 INFO][tester.py:150 - log_eval_epoch_info] -
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 40000   | 1.01375 | 0.01021 | 0.01205 | 0.00028 |
+---------+---------+---------+---------+---------+
```


</p>
</details>  

## Citation
```
@inproceedings{park2023acls,
  title={{ACLS}: Adaptive and conditional label smoothing for network calibration},
  author={Park, Hyekang and Noh, Jongyoun and Oh, Youngmin and Baek, Donghyeon and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## References
Our work is mainly built on [FLSD](https://github.com/torrvision/focal_calibration), [MbLS](https://github.com/by-liu/MbLS), and [CALS](https://github.com/by-liu/CALS). Thanks to the authors!

