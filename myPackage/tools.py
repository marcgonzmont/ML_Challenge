from os import makedirs, errno, altsep
from os.path import exists

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