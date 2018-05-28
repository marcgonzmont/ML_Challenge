from os import makedirs, errno
from os.path import exists, join
import numpy as np
from matplotlib import pyplot as plt
import itertools
from sklearn import preprocessing

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix
    :param classes: array of classes' names
    :param normalize: boolean
    :param title: plot title
    :param cmap: colour of matrix background
    :return: plot confusion matrix
    '''

    # plt_name = altsep.join((plot_path,"".join((title,".png"))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    print('\nSum of main diagonal')
    print(np.trace(cm))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=0)

    # plt.savefig(plt_name)
    plt.show()


def normalize(data):
    '''
    Normalize input data [0, 1]
    :param data: input data
    :return: normalized data
    '''
    scaler = preprocessing.MinMaxScaler()
    data_min_max = scaler.fit_transform(data)
    return data_min_max
