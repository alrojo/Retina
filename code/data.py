import glob 
from sys.path import insert
from os.path import join, getcwd
import pandas as pd 

# Hallihula
insert(0, os.getcwd)
import paths
from sklearn.cross_validation import train_test_split

N_images = main.N_images ## TODO(1)


#This has been moved to main
# class Paths(object):	
# 	data_dir = join(getcwd(),'data'))
# 	train_dir = join(data_dir,'train')
# 	test_dir = train = join(data_dir,'test')
# 	train_labels = join(data_dir,'trainLabels.csv')
# 	#X_train should be added in data.py
# 	#X_valid should be added in data.py
# 	#X_test should be added in data.py



#train_images = glob.glob(os.path.join(Paths.train,"*")
# I don't need the above because all the filenames are already in trainlabel, plus their labels 

train_image_names = pd.read_csv(Paths.train_labels)['image']
train_labels = pd.read_csv(Paths.train_labels)['level']
N = len(train_labels)

try: 
    assert( (N_images == 'ALL') or ( isinstance(N_images, int) and (N_images <= N) and (N_images > 0) ) )
except AssertionError:
    print "N_images specified wrong setting N_images = 'ALL' = %d\n" % N
    N_images = N


X_train_names, X_valid_names, y_train_labels, y_valid_labels = train_test_split(
	train_image_names, train_labels, test_size = 0.40, random_state=42)  #42 is the meaning life and everything

# I wonder if this is now available for all instances of this class?
Paths.X_train = np.array[join(Paths.train_dir, "%s.jpeg" % image) for image in X_train_names]
Paths.X_valid = np.array[join(Paths.train_dir, "%s.jpeg" % image) for image in X_valid_names]

## TODO check that the sort is correct.
Paths.X_test = (glob.glob(os.path.join(Paths.test,"*"))).sort()

# 








