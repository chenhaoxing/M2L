import os
import csv
import numpy as np
import random
from PIL import Image
import pdb


data_dir = '/home/data/TieredImageNet'                # the path of the download dataset
save_dir = '/home/data//TieredImageNet'    # the saving path of the divided dataset


if not os.path.exists(os.path.join(save_dir, 'images')):
	os.makedirs(os.path.join(save_dir, 'images'))



Train_images_dir = os.path.join(data_dir, 'train')
Val_images_dir = os.path.join(data_dir, 'val')
Test_images_dir = os.path.join(data_dir, 'test')



# save data into csv file----- Train
train_data = []
for class_name in os.listdir(Train_images_dir):
	class_dir = os.path.join(Train_images_dir, class_name)

	images = [[i, class_name] for i in os.listdir(os.path.join(class_dir))]
	train_data.extend(images)
	print('Train----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(class_dir, i) for i in os.listdir(os.path.join(class_dir))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)


with open(os.path.join(save_dir, 'train.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(train_data)




# save data into csv file----- Val
val_data = []
for class_name in os.listdir(Val_images_dir):
	class_dir = os.path.join(Val_images_dir, class_name)

	images = [[i, class_name] for i in os.listdir(os.path.join(class_dir))]
	val_data.extend(images)
	print('Val----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(class_dir, i) for i in os.listdir(os.path.join(class_dir))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)


with open(os.path.join(save_dir, 'val.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(val_data)





# save data into csv file----- Test
test_data = []
for class_name in os.listdir(Test_images_dir):
	class_dir = os.path.join(Test_images_dir, class_name)

	images = [[i, class_name] for i in os.listdir(os.path.join(class_dir))]
	test_data.extend(images)
	print('Test----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(class_dir, i) for i in os.listdir(os.path.join(class_dir))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)


with open(os.path.join(save_dir, 'test.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(test_data)