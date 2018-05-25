from os import makedirs, errno
from os.path import exists, join
import numpy as np

def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        if not exists(path):
            makedirs(path)
            print("\nCreated '{}' folder\n".format(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = [data[i] for i in train_indices]
    test_set  = [data[i] for i in test_indices]
    return np.asarray(train_set), np.asarray(test_set)
