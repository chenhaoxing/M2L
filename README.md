# Multi-level Metric Learning for Few-Shot Image Recognition
This code implements the Multi-level Metric Learning for Few-shot Image Recognition (M2L).


## Citation
If you find our work useful, please consider citing our work using the bibtex:
```
@inproceedings{chen2022multi,
	author  = {Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
	title   = {Multi-level Metric Learning for Few-Shot Image Recognition},
	booktitle = {International Conference on Artificial Neural Networks(ICANN)},
	year    = {2022},
}
```


## Prerequisites
* Linux
* Python 3.7
* Pytorch 1.0+
* GPU + CUDA CuDNN
* pillow, torchvision, scipy, numpy

## Datasets
**Dataset download link:**
* [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view) It contains 100 classes with 600 images in each class, which are built upon the ImageNet dataset. The 100 classes are divided into 64, 16, 20 for meta-training, meta-validation and meta-testing, respectively.
* [tieredImageNet](https://drive.google.com/drive/folders/163HGKZTvfcxsY96uIF6ILK_6ZmlULf_j?usp=sharing)
TieredImageNet is also a subset of ImageNet, which includes 608 classes from 34 super-classes. Compared with miniImageNet, the splits of meta-training(20), meta-validation(6) and meta-testing(8) are set according to the super-classes to enlarge the domain difference between training and testing phase. The dataset also include more images for training and evaluation (779,165 images in total).
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
CIFAR-FS is divided from CIFAR-100, which consists of 60,000 images in 100 categories. The CIFAR-FS is divided into 64, 16 and 20 for training, validation, and evaluation, respectively.
* [FC-100](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing)
FC-100 is also divided from CIFAR-100, which is more difficult because it is more diverse. The FC100 uses a split similar to tieredImageNet, where train, validation, and test splits contain 60, 20, and 20 classes.


**Note: You need to manually change the dataset directory.**

## Pre-trained backbone
We provide pre-trained backbones at https://pan.baidu.com/s/1v2k-mdCpGLtKnKG5ijYXMw  keys: 334q

## Few-shot Classification
* Train a 5-way 1-shot MML model based on ResNet-12 (on miniImageNet dataset):
```
 python experiments/run_trainer.py  --cfg ./configs/miniImagenet/MML_N5K1_R12.yaml --device 0
```
Test model on the test set:
```
python experiments/run_evaluator.py --cfg ./configs/miniImagenet/MML_N5K1_R12.yaml -c ./checkpoint/*/*.pth --device 0
```

## Contacts
Please feel free to contact us if you have any problems.

Email: haoxingchen@smail.nju.edu.cn
