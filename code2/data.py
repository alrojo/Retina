#from sys.path import insert
import glob 
import os
import pandas as pd 

# Hallihula
#insert(0, os.getcwd)
from cfg import paths
from sklearn.cross_validation import train_test_split

#N_images = main.N_images ## TODO(1)


# TODO: I need to look at line 9 in data.py (benanne).  He has the data/split.pkl
#train_images = glob.glob(os.path.join(Paths.train,"*")
# I don't need the above because all the filenames are already in trainlabel, plus their labels 


def data():
    """
    It is what it is
    """
    train_image_names = pd.read_csv(paths.train_labels)['image']
    train_labels = pd.read_csv(paths.train_labels)['level']
    N = len(train_labels)

    try:
        assert((N_images == 'ALL') or (isinstance(N_images, int) and (N_images <= N) and (N_images > 0)))
    except AssertionError:
        print "N_images specified wrong setting N_images = 'ALL' = %d\n" % N
        N_images = N


    X_train_names, X_valid_names, y_train_labels, y_valid_labels = train_test_split(
    	train_image_names, train_labels, test_size = 0.40, random_state=42)  #42 is the meaning life and everything

    # I wonder if this is now available for all instances of this class?
    paths.X_train = np.array[os.path.join(paths.train_dir, "%s.jpeg" % image) for image in X_train_names]
    paths.X_valid = np.array[os.path.join(paths.train_dir, "%s.jpeg" % image) for image in X_valid_names]

    ## TODO check that the sort is correct.
    paths.X_test = (glob.glob(os.path.join(paths.test,"*"))).sort()


    # TODO: fix this notation thing.  For now I'm reassingin variables to bennane notation
    paths_my_train = paths.X_train
    paths_my_valid = paths.X_valid
    labels_my_valid = paths.X_valid



def gen_images(paths, labels=None, shuffle=False, repeat=False):
    paths_shuffled = np.array(paths)

    if labels is not None:
        labels_shuffled = np.array(labels)

    while True:
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(paths_shuffled)
            if labels is not None:
                np.random.set_state(state)
                np.random.shuffle(labels_shuffled)

        for k in xrange(len(paths_shuffled)):
            path = paths_shuffled[k]
            im = skimage.io.imread(os.path.join("data", path))
            im = im.transpose(2, 0, 1).astype('float32') / 255.0

            if labels is not None:
                yield im, labels_shuffled[k]
            else:
                yield im
        
        if not repeat:
            break


def gen_chunks(image_gen, chunk_size=8192, labels=True):
    chunk = np.zeros((chunk_size, 3, 96, 96), dtype='float32')
    
    if labels:
        chunk_labels = np.zeros((chunk_size, 1), dtype='float32')

    offset = 0

    for sample in image_gen:
        if labels:
            im, label = sample
        else:
            im = sample

        chunk[offset] = im

        if labels:
            chunk_labels[offset] = label

        offset += 1

        if offset >= chunk_size:
            if labels:
                yield chunk, chunk_labels, offset
            else:
                yield chunk, offset

            chunk = np.zeros((chunk_size, 3, 96, 96), dtype='float32')

            if labels:
                chunk_labels = np.zeros((chunk_size, 1), dtype='float32')

            offset = 0

    if offset > 0:
        if labels:
            yield chunk, chunk_labels, offset
        else:
            yield chunk, offset
#



if __name__ == '__main__':
    import fun
    data()
    #fun.usage()
    #exit()




