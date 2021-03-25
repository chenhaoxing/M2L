import os
import sys
import time
import tqdm
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchnet as tnt
import torchvision.transforms as transforms
sys.dont_write_bytecode = True

from dataset.BaseDataSet import FewShotDataSet


# Batch(list terms:5):
# support_images: 8x5x3x84x84
# support_labels: 8x5
# query_images: 8x30x3x84x84
# query_labels: 8x30
# 8x5
# 8



def get_dataset(opt, mode):
	assert mode in ['train', 'val', 'test']

	if opt.augment and mode == 'train':

		transform = transforms.Compose([
			transforms.Resize((100, 100)),
			transforms.RandomCrop(opt.imageSize),
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

	else:

		transform = transforms.Compose([
			transforms.Resize((opt.imageSize, opt.imageSize)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	dataset = FewShotDataSet(opt, transform, mode)

	return dataset




def get_dataloader(opt, modes):

	loaders = []
	for mode in modes:
		dataset = get_dataset(opt, mode)

		if mode == 'train':
			loader = FewShotDataloader(dataset, batch_size=opt.episodeSize, epoch_size=opt.episode_train_num, shuffle=True,
								num_workers=int(opt.workers), current_epoch=opt.current_epoch)
		elif mode == 'val':
			loader = FewShotDataloader(dataset, batch_size=opt.testepisodeSize, epoch_size=opt.episode_val_num, shuffle=False,
								num_workers=int(opt.workers), current_epoch=opt.current_epoch)
		elif mode == 'test':
			loader = FewShotDataloader(dataset, batch_size=opt.testepisodeSize, epoch_size=opt.episode_test_num, shuffle=False,
								num_workers=int(opt.workers), current_epoch=opt.current_epoch)
		else:
			raise ValueError('Mode ought to be in [train, val, test]')

		loaders.append(loader)

	return loaders




class FewShotDataloader(object):
	def __init__(self,
				 dataset,
				 batch_size=1,
				 epoch_size=2000,
				 shuffle=True,
				 num_workers=4,
				 current_epoch=0
				 ):
		self.dataset     = dataset
		self.data_root   = dataset.data_root  # the root file to store images
		self.way_num     = dataset.way_num
		self.shot_num    = dataset.shot_num
		self.query_num   = dataset.query_num
		self.transform   = dataset.transform
		self.loader      = dataset.loader

		self.batch_size  = batch_size
		self.epoch_size  = epoch_size
		self.num_workers = num_workers
		self.shuffle     = shuffle
		self.current_epoch  = current_epoch

	def sampleImageIdsFrom(self, cat_id, sample_size=1):
		assert (cat_id in self.dataset.class_img_dict)
		assert (len(self.dataset.class_img_dict[cat_id]) >= sample_size)
		# Note: random.sample samples elements without replacement.
		return random.sample(self.dataset.class_img_dict[cat_id], sample_size)

	def sampleCategories(self, sample_size=1):
		class_list = self.dataset.class_list
		assert (len(class_list) >= sample_size)
		return random.sample(class_list, sample_size)

	def sample_query_examples(self, categories, query_num):
		Tbase = []
		if len(categories) > 0:
			# average sample (keep the number of images each category)
			for K_idx, K_cat in enumerate(categories):
				img_ids = self.sampleImageIdsFrom(
					K_cat, sample_size=query_num)
				Tbase += [(img_id, K_idx) for img_id in img_ids]

		assert(len(Tbase)) == len(categories) * query_num
		return Tbase

	def sample_support_and_query_examples(
			self, categories, query_num, shot_num):
		if len(categories) == 0:
			return [], []
		nCategories = len(categories)
		Query_imgs = []
		Support_imgs = []

		for idx in range(len(categories)):
			img_ids = self.sampleImageIdsFrom(
				categories[idx],
				sample_size=(query_num + shot_num)
			)
			imgs_novel = img_ids[:query_num]
			imgs_exemplar = img_ids[query_num:]

			Query_imgs += [(img_id, idx) for img_id in imgs_novel]
			Support_imgs += [(img_id, idx) for img_id in imgs_exemplar]

		assert(len(Query_imgs) == nCategories * query_num)
		assert(len(Support_imgs) == nCategories * shot_num)

		return Query_imgs, Support_imgs

	def sample_episode(self):
		"""
			Samples a training episode.
		"""
		way_num  = self.way_num
		shot_num  = self.shot_num
		query_num = self.query_num
		categories = self.sampleCategories(way_num)
		Query_imgs, Support_imgs = self.sample_support_and_query_examples(categories, query_num, shot_num)

		return Query_imgs, Support_imgs

	def createExamplesTensorData(self, examples):
		images = torch.stack([self.transform(self.loader(os.path.join(self.data_root, 'images', img_name))) for img_name, _ in examples], dim=0)
		labels = torch.tensor([label for _, label in examples]).long()

		return images, labels

	def get_iterator(self, index):
		rand_seed = index
		random.seed(rand_seed)
		np.random.seed(rand_seed)

		def load_function(iter_idx):
			Query_imgs, Support_imgs = self.sample_episode()
			Xt, Yt = self.createExamplesTensorData(Query_imgs)
			Xe, Ye = self.createExamplesTensorData(Support_imgs)
			return Xt, Yt, Xe, Ye

		tnt_dataset = tnt.dataset.ListDataset(
			elem_list=range(self.epoch_size), load=load_function)
		data_loader = tnt_dataset.parallel(
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=self.shuffle)

		return data_loader

	def __call__(self, index):
		return self.get_iterator(index)

	def __len__(self):
		return int(self.epoch_size / self.batch_size)
