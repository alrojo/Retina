import os
"""See How do I share global variables across modules? at 
https://docs.python.org/2/faq/programming.html#what-are-the-rules-for-local-and-global-variables-in-python
"""


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


paths = PathLibrary
