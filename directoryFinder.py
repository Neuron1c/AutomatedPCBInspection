from os import listdir
from os.path import isfile, join

def getLastCount(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return len(onlyfiles) + 1

def getList(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles
