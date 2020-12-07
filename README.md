# Multi-level Metric Learning for Few-Shot Image Recognition
This code implements the Multi-level Metric Learning for Few-shot Image Recognition (M2L).

Our code will be released soon.

## Citation
If you find our work useful, please consider citing our work using the bibtex:
```
@Article{chen2020multi,
	author  = {Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
	title   = {Multi-level Metric Learning for Few-Shot Image Recognition},
	journal = {arXiv preprint arXiv:...},
	year    = {2020},
}
```

## Prerequisites
* Linux
* Python 3.6
* Pytorch 1.0+
* GPU + CUDA CuDNN
* pillow, torchvision, scipy, numpy

## Datasets
**Dataset download link:**
* [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view)

**Note: You need to manually change the dataset directory.**

## miniImageNet Few-shot Classification
* Train a 5-way 1-shot model based on Conv-64F:
```
python M2L_Train_5way1shot.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet
```
Test model on the test set:
```
python M2L_Test_5way1shot.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --resume 
```
## Contacts
Please feel free to contact us if you have any problems.

Email: haoxingchen@smail.nju.edu.cn
