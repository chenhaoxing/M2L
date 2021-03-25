import os
import os.path as path
import numpy as np
from PIL import Image
import csv
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import sys
import warnings
warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True




def load_csv2dict(csv_path):
	class_img_dict = {}
	with open(csv_path) as csv_file:
		csv_context = csv.reader(csv_file, delimiter=',')
		for line in csv_context:
			if csv_context.line_num == 1:
				continue
			img_name, img_class = line

			if img_class in class_img_dict:
				class_img_dict[img_class].append(img_name)
			else:
				class_img_dict[img_class] = []
				class_img_dict[img_class].append(img_name)
	class_list = class_img_dict.keys()
	return class_img_dict, class_list



def read_csv(filePath):
	'''
	Read file name and it's label
	:param filePath:
	:return:
	'''
	imgs = []
	with open(filePath, 'r') as csvFile:
		lines = csvFile.readlines()
		for line in lines:
			context = line.split(',')
			try:
				imgs.append((context[0].strip(), context[1].strip()))
			except:
				print('Escape context:{}'.format(context))
	return imgs


def PIL_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')


def Default_loader(path):
	return io.imread(path)


def RGB_loader(path):
	return Image.open(path).convert('RGB')


def Gray_loader(path):
	return Image.open(path).convert('L')



class FewShotDataSet(Dataset):
	def __init__(self, opt, transform=None, phase='train', loader=RGB_loader):
		super(FewShotDataSet, self).__init__()

		self.phase = phase
		self.transform = transform
		self.loader = loader
		self.way_num = opt.way_num
		self.shot_num = opt.shot_num
		self.query_num = opt.query_num
		self.data_root = opt.dataset_dir

		assert (phase in ['train', 'val', 'test'])

		print('Loading dataset - phase {0}'.format(phase))
		if phase == 'train':
			self.csv_path    = os.path.join( self.data_root, 'train.csv')
		elif phase == 'test':
			self.csv_path    = os.path.join( self.data_root, 'test.csv')
		elif phase == 'val':
			self.csv_path    = os.path.join( self.data_root, 'val.csv')
		else:
			raise ValueError('phase ought to be in [train/test/val]')

		self.data_list = read_csv(self.csv_path)
		self.class_img_dict, class_list = load_csv2dict(self.csv_path)
		self.class_list = sorted(list(class_list))
		self.label2Int = {item: idx for idx, item in enumerate(self.class_list)}
		self.num_cats = len(self.class_list)
		pass

	def __getitem__(self, index):
		fn, class_name = self.data_list[index]
		label = self.label2Int[class_name]
		img = self.loader(fn)
		if self.transform is not None:
			img = self.transform(img)
		return img, label


