# Multi-level Metric Learning for Few-Shot Image Recognition
This code implements the Multi-level Metric Learning for Few-shot Image Recognition (M2L).


## Citation
If you find our work useful, please consider citing our work using the bibtex:
```
@Article{chen2021multi,
	author  = {Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
	title   = {Multi-level Metric Learning for Few-Shot Image Recognition},
	journal = {arXiv preprint arXiv:2103.11383},
	year    = {2021},
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
* [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view)
* [tieredImageNet](https://drive.google.com/drive/folders/163HGKZTvfcxsY96uIF6ILK_6ZmlULf_j?usp=sharing)
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [Stanford Dog](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [Stanford Car](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

**Note: You need to manually change the dataset directory.**

## Few-shot Classification
* Train a 5-way 1-shot MML(KL) model based on Conv-64F (on miniImageNet dataset):
```
python MML_Train_1shot.py --method_name KL --dataset_dir ./datasets/miniImageNet --data_name miniImageNet
```
Test model on the test set:
```
python Test_Batch.py --method_name KL --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --resume ./results/miniImageNet_KL_1shot\KL_BatchSize_4_Conv64F_miniImageNet_5Way_1Shot/model_best_test.pth.tar 
```


## Pre-trained models
We also provide some pre-trained models.
You can run the follow command to evaluate the model
```
python Test_Batch.py --method_name Wass --dataset_dir ./datasets/tieredImageNet --data_name tieredImageNet --resume ./results/tieredImageNet_Wass_1shot\Wass_BatchSize_4_Conv64F_tieredImageNet_5Way_1Shot/model_best_test.pth.tar 
```
## Contacts
Please feel free to contact us if you have any problems.

Email: haoxingchen@smail.nju.edu.cn
