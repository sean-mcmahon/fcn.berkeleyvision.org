#! /usr/bin/python
"""
Created on Wed Aug 17 2016
Generated text files containing file paths for different data-subsets
(cross-validation of the data)
@author: Sean McMahon
"""
import numpy as np
import argparse
import glob
import random
import os.path

# Input:
# - training directories e.g. .../12Aug/2016-08-12-10-09-26_groundfloorCarPark
# - testing directories
# - validation percentage of training directory
#    + Random Sample OR Sample Beginning or End of LABELLED data,
#
# Output:
# train/test/val .txt containing lines with:
# working_dir/index
# index = xxxx-x_xxx...x
# e.g. 0000-5_1470962221647243797

def getImgIndex(filename):
    # sample string: colourimg_0459-5_1470961026872523873.png
    return filename[filename.find('_')+1:filename.rfind('_')]

def getSubSetName(filename):
    last_slash = filename.rfind('/')
    if last_slash==(len(filename)-1):
        # If I accidently put slash at end of filename
        last_slash = filename.rfind('/',end=len(filename)-2)
        # return dir name without trailing slash
        return filename[last_slash+1:-1]
    return filename[last_slash+1:]

parser = argparse.ArgumentParser()
parser.add_argument('--training_dir', nargs='+', help='List of directories')
parser.add_argument('--testing_dir', default='')
parser.add_argument('--val_percentage', default=0.15)
parser.add_argument('--out_dir', default='/home/sean/hpc-home/Construction_Site/Springfield/12Aug16')
args = parser.parse_args()

# Get train directory and indices
train_dir_index = []; test_dir_index = []
for train_path in args.training_dir:
    current_training_dir = getSubSetName(train_path)
    img_fullnames = glob.glob(train_path+'/colour/*.png')
    for image_name in img_fullnames:
        img_index = getImgIndex(os.path.basename(image_name))
        train_dir_index.append([current_training_dir, img_index])

# Get test directory and indecies
for test_path in args.testing_dir:
    current_testing_dir = getSubSetName(test_path)
    img_fullnames = glob.glob(test_path+'colour/*.png')
    for image_name in img_fullnames:
        img_index = getImgIndex(os.path.basename(image_name))
        test_dir_index.append([current_testing_dir, img_index])

# create val directory and indecies list
num_val_instances = int( (args.val_percentage * len(train_dir_index)) /2)
val_dir_index = train_dir_index[0:num_val_instances]
val_dir_index.append(train_dir_index[-num_val_instances:])
train_dir_index[0:num_val_instances] = []
train_dir_index[-num_val_instances:] = []

testfile = open(args.out_dir + '/test.txt','w')
for item in test_dir_index:
    testfile.write('%s %s\n',item[0],item[1] )

print 'train list has {} elements, sub-element length {}\nExample sub-element: {}'.format(len(train_dir_index),len(train_dir_index[-1]), train_dir_index[-1])
