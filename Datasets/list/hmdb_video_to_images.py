import pandas as pd
import numpy as np
from PIL import Image
import skimage
import imageio
import os
import glob
import warnings
import natsort

warnings.filterwarnings('ignore')

def findFiles(path):
	return natsort.natsorted(glob.glob(path))

testlist = findFiles('three splits/split1/*.txt')
trainsample = []
trainlabel = []
testsample = []
testlabel = []
categories = []
for i in range(len(testlist)):
	filename = testlist[i].split('/')[-1].split('_test_')[0]
	categories.append(filename)
	listall = pd.read_table(testlist[i], header=None, delim_whitespace=True)
	for j in range(len(listall)):
		trainsample.append(listall.ix[j, 0])
		trainlabel.append(filename)


for idx in range(len(trainlabel)):
	video_dir = os.path.join('HMDB-51', trainlabel[idx], trainsample[idx])
	video = imageio.get_reader(video_dir, 'ffmpeg')
	print('the training number is: ', idx)
	sample = trainsample[idx].split('.')[0]
	sample_dir = os.path.join('HMDB', trainlabel[idx], sample)
	os.makedirs(sample_dir)
	for num, img in enumerate(video):
		image = skimage.img_as_float(img).astype(np.float32)
		image = Image.fromarray(np.uint8(image * 255))
		image.save(sample_dir +  '/' + str(num) + '.jpg')
	if len(findFiles(sample_dir + '/*.jpg')) == 0:
		print(sample_dir)
		break

for idx in range(len(testlabel)):
	video_dir = os.path.join('HMDB-51', testlabel[idx], testsample[idx])
	video = imageio.get_reader(video_dir, 'ffmpeg')
	print('the testing number is: ', idx)
	sample = testsample[idx].split('.')[0]
	sample_dir = os.path.join('HMDB', testlabel[idx], sample)
	os.makedirs(sample_dir)
	for num, img in enumerate(video):
		image = skimage.img_as_float(img).astype(np.float32)
		image = Image.fromarray(np.uint8(image * 255))
		image.save(sample_dir +  '/' + str(num) + '.jpg')
	if len(findFiles(sample_dir + '/*.jpg')) == 0:
		print(sample_dir)
		break