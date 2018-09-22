import pandas as pd
import numpy as np
from PIL import Image
import skimage
import imageio
import os
import glob
import warnings

warnings.filterwarnings('ignore')
def findFiles(path):
	return glob.glob(path)

trainlist = pd.read_table('ucfTrainTestlist/trainlist01.txt', header=None, delim_whitespace=True)
testlist = pd.read_table('ucfTrainTestlist/testlist01.txt', header=None, delim_whitespace=True)


for idx in range(len(trainlist)):
	print('the training number is: ', idx)
	video_dir = os.path.join('UCF-101', trainlist.ix[idx, 0])
	video = imageio.get_reader(video_dir, 'ffmpeg')
	sample_dir = os.path.join('UCF', trainlist.ix[idx, 0].split('.')[0])
	os.makedirs(sample_dir)
	for num, img in enumerate(video):
		image = skimage.img_as_float(img).astype(np.float32)
		image = Image.fromarray(np.uint8(image * 255))
		image.save(sample_dir +  '/' + str(num) + '.jpg')

for idx in range(len(testlist)):
	print('the testing number is: ', idx)
	video_dir = os.path.join('UCF-101', testlist.ix[idx, 0])
	video = imageio.get_reader(video_dir, 'ffmpeg')
	sample_dir = os.path.join('UCF', testlist.ix[idx, 0].split('.')[0])
	os.makedirs(sample_dir)
	for num, img in enumerate(video):
		image = skimage.img_as_float(img).astype(np.float32)
		image = Image.fromarray(np.uint8(image * 255))
		image.save(sample_dir +  '/' + str(num) + '.jpg')

