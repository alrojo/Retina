

from os import getcwd
from os.path import join
import pandas as pd 

#How many images to train on 
N_images = 100 #integer or ALL

class PathLibrary(object):
 	data_dir = join(getcwd(),'../data')
 	train_dir = join(data_dir,'train')
 	sample_dir = join(data_dir,'sample')
 	test_dir = train = join(data_dir,'test')
 	train_labels = join(data_dir,'trainLabels.csv')

 	#X_train should be added in data.py
 	#X_valid should be added in data.py
 	#X_test should be added in data.py


paths = PathLibrary

