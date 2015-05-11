# import time
import os
import sys
# import pandas as pd
# import numpy as np
# import theano
# import theano.tensor as T
# import lasagne as lag
import importlib
from memory_profiler import profile as memory_profiling
import getopt
import fun
import train
from cfg import paths

# Use this when you start to keep a log

# to make thdir


class PathLibrary(object):
            data_dir = os.path.join(os.getcwd(), '../data')
            metadata_dir = os.path.join(data_dir, 'metadata')
            train_dir = os.path.join(data_dir, 'train')
            sample_dir = os.path.join(data_dir, 'sample')
            test_dir = train = os.path.join(data_dir, 'test')
            train_labels = os.path.join(data_dir, 'trainLabels.csv')

#             X_train should be added in data.py
#             X_valid should be added in data.py
#             X_test should be added in data.py


@memory_profiling
def program():
    """This should handle everything"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'N:c:', ['help'])
        assert(not args)
    except AssertionError:
        print "Arguments supplied wrong."
        fun.usage()
        sys.exit(2)
    except getopt.GetoptError as err:
        print (err)
        fun.usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-N":
            N_images = a
        elif o == "-c":
            config_name = str(a)
        elif o == "--help":
            fun.usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    try:
        N_images
        print "Training on %s samples" % str(N_images)
    except NameError:
        print "setting N_images to 'all' "
        N_images = "all"
    try:
        config_name
    except NameError:
        print "You need to specify a config file"
        fun.usage()
        sys.exit()
    fun.usage()
    paths = PathLibrary
    config = importlib.import_module("configurations.%s" % config_name)
    train.train(config=config, N_train=N_images,
                exp_id=config_name, paths=paths)


if __name__ == '__main__':
    program()