#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Haoxingchen (haoxingchen@smail.nju.edu.cn)
Date: March 16, 2021
Version: V0

Citation:
@inproceedings{chen2021multi,
  title={Multi-level Metric Learning for Few-Shot Image Recognition},
  author={Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
  journal={arXiv preprint arXiv:2103.11383,
  year={2021}
}
"""


from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from torch import autograd
from PIL import ImageFile
import pdb
import sys
sys.dont_write_bytecode = True


# ============================ Data & Networks =====================================
import dataset.few_shot_dataloader as FewShotDataloader
from models import network_M2L_KL,network_M2L_Wass
import utils
model_dict = dict(
    KL=network_M2L_KL,
    Wass = network_M2L_Wass)
# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='5'



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./datasets/TieredImageNet', help='/miniImageNet')
parser.add_argument('--data_name', default='TieredImageNet', help='miniImageNet | tieredImageNet | CUB | StanfordDog | StanfordCar')
parser.add_argument('--method_name', default='Wass', help=' Wass |  KL ')
parser.add_argument('--mode', default='test', help='train|val|test')
parser.add_argument('--outf', default='/home/chenhaoxing/MANet/results/TieredImageNet_EMD_5shot/EMD_BatchSize_4_Conv64F_TieredImageNet_5Way_5Shot')
parser.add_argument('--resume', default='/home/chenhaoxing/MANet/results/TieredImageNet_EMD_5shot/EMD_BatchSize_4_Conv64F_TieredImageNet_5Way_5Shot/model_best_test.pth.tar', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--basemodel', default='Conv64F', help='Conv64F')
parser.add_argument('--workers', type=int, default=8)
#  Few-shot parameters  #
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--augment', action='store_true', default=True, help='Perform data augmentation or not')
parser.add_argument('--episodeSize', type=int, default=4, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=4, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=50, help='the total number of training epoch')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=5, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--neighbor_k', type=int, default=1, help='the number of k-nearest neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.005')
parser.add_argument('--adam', action='store_true', default=True, help='use adam optimizer')
parser.add_argument('--cosine', type=bool, default=False, help='using cosine annealing')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True



# ======================================= Define functions =============================================
def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
  

	# switch to evaluate mode
	model.eval()
	accuracies = []


	end = time.time()
	for episode_index, (batch) in enumerate(val_loader(epoch_index)):

		query_images, query_targets, support_images, support_targets = [item.cuda() for item in batch]
		
		"""
		query_images:    (4, 75, 3, 84, 84)       ==> (batch_size, query_nums * way_nums, channels, H, W)
		query_targets:   (4, 75)                  ==> (batch_size, query_nums * way_nums)
		support_images:  (4, 5, 3, 84, 84)        ==> (batch_size, shot_nums * way_nums, channels, H, W)
		support_targets: (4, 5)                   ==> (batch_size, shot_nums * way_nums)
		"""

		batch_size = support_targets.shape[0]
		support_nums = support_targets.shape[-1]
		query_nums = query_targets.shape[-1] 


		# Convert query and support images
		input_var1 = query_images.contiguous().view(-1, query_images.size(2), query_images.size(3), query_images.size(4))
		input_var2 = support_images.contiguous().view(-1, support_images.size(2), support_images.size(3), support_images.size(4))


		# Calculate the output
		output, Q_S = model(input_var1, input_var2)


		# Calculate the losses
		loss = torch.tensor(0.).cuda()
		loss.requires_grad = True
		for i in range(len(output)):
			temp_loss = criterion(output[i], query_targets[i])
			loss = loss + temp_loss

			# Measure accuracy and record loss
			prec1, _ = utils.accuracy(output[i], query_targets[i], topk=(1,3))
			losses.update(temp_loss.item(), query_nums)
			top1.update(prec1[0], query_nums)
			accuracies.append(prec1)


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1), file=F_txt)

		
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)

	return top1.avg, losses.avg, accuracies




if __name__=='__main__':

	# save path
	opt.outf, F_txt = utils.set_save_path(opt)

	FewShotNet = model_dict[opt.method_name]
	model = FewShotNet.define_FewShotNet(which_model=opt.basemodel, num_classes=opt.way_num, neighbor_k=opt.neighbor_k, norm='batch', 
		shot_num=opt.shot_num, batch_size=opt.episodeSize, init_type='normal', use_gpu=opt.cuda)

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	# ============================================ Test phase ============================================
	# Set the save path
	F_txt_test = utils.set_save_test_path(opt)
	print('========================================== Start Test ==========================================\n')
	print('========================================== Start Test ==========================================\n', file=F_txt_test)

	# Load the trained best model
	best_model_path = os.path.join(opt.outf, 'model_best_test.pth.tar')
	checkpoint = torch.load(opt.resume)
	epoch_index = checkpoint['epoch_index']
	best_prec1 = checkpoint['best_prec1_test']
	model.load_state_dict(checkpoint['state_dict'])


	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt_test)
	print(model) 
	print(model, file=F_txt_test) 


	# Repeat five times
	repeat_num = 5       
	total_accuracy = 0.0
	total_h = np.zeros(repeat_num)
	for r in range(repeat_num):
		
		print('==================== The %d-th round ====================' %r)
		print('==================== The %d-th round ====================' %r, file=F_txt_test)

		# ======================================= Loaders of Datasets =======================================
		opt.current_epoch = r
		_, _, test_loader = FewShotDataloader.get_dataloader(opt, ['train', 'val', 'test'])


		# evaluate on validation/test set
		prec1, test_loss, accuracies = validate(test_loader, model, criterion, epoch_index, best_prec1, F_txt_test)
		test_accuracy, h = utils.mean_confidence_interval(accuracies)
		total_accuracy += test_accuracy
		total_h[r] = h

		print('Test accuracy: %f h: %f \n' %(test_accuracy, h))
		print('Test accuracy: %f h: %f \n' %(test_accuracy, h), file=F_txt_test)
	print('Mean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()))
	print('Mean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()), file=F_txt_test)
	print('===================================== Test is END =====================================\n')
	print('===================================== Test is END =====================================\n', file=F_txt_test)
	F_txt_test.close()