import os
import sys
import time
import numpy as np
import codecs
import tensorflow as tf
import xlrd

def load_data(folders):
    """
        Load all the images in the folder
    """
    examples = []
    count=0
    pathstring=[]
    for folder in folders:
        for f in os.listdir(folder):
            count+=1
            print(count)
            path=folder+f
            pathstring.append(path.strip())
    return pathstring

def main(args):
    data_dir=['..//qd_data//']
    imagename=load_data(data_dir)
    perm = np.arange(len(imagename))
    np.random.shuffle(perm)
    imagename = np.asarray(imagename)
    train_data = imagename[perm]
    print(len(train_data))
    print(train_data[0:20])
    with codecs.open("..//dataset//imgpath.txt",'w',encoding='utf-8') as f:
        for name in train_data:
            f.write(name)
            f.write('\n')
        f.close()
if __name__=='__main__':
    main(sys.argv)
    # imagefiles=[]
    # with codecs.open("imagename.txt",'r',encoding='utf-8') as file:
    #     line = file.readline()
    #     while line:
    #         imagefiles.append(line.strip())
    #         line = file.readline()
