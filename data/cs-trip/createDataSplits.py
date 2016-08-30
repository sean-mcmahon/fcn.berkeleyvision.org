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
    # sample string: colourimg_0459-5_1470961026872523873_objects.mat
    return filename[filename.find('_') + 1:filename.rfind('_', 0, filename.rfind('_'))]


def getSubSetName(filename):
    last_slash = filename.rfind('/')
    if last_slash == (len(filename) - 1):
        # If I accidently put slash at end of filename
        last_slash = filename.rfind('/', end=len(filename) - 2)
        # return dir name without trailing slash
        return filename[last_slash + 1:-1]
    return filename[last_slash + 1:]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_directory',
                    default='/home/sean/hpc-home/Construction_Site/Springfield/12Aug16/K2')
parser.add_argument('--training_dir', nargs='+', help='List of directories')
parser.add_argument('--testing_dir', nargs='+')
parser.add_argument('--val_percentage', default=0.15)
parser.add_argument(
    '--out_dir', default='/home/sean/Dropbox/Uni/Code/FCN_models/data/cs-trip')
args = parser.parse_args()
training_dir = args.training_dir
testing_dir = args.testing_dir

if testing_dir in training_dir:
    print 'WARNING: same data subsets used in training and testing'

# Get train directory and indices
print 'Getting train directory and indecies'
train_dir_index = []
test_dir_index = []
for train_path in training_dir:
    current_training_dir = getSubSetName(train_path)
    img_fullnames = glob.glob(
        args.dataset_directory + '/' + train_path + '/labels/*.mat')
    for image_name in img_fullnames:
        img_index = getImgIndex(os.path.basename(image_name))
        train_dir_index.append([current_training_dir, img_index])

# Get test directory and indecies
print 'Getting test directory and indecies'
for test_path in testing_dir:
    current_testing_dir = getSubSetName(test_path)
    img_fullnames = glob.glob(
        args.dataset_directory + '/' + test_path + '/labels/*.mat')
    for image_name in img_fullnames:
        img_index = getImgIndex(os.path.basename(image_name))
        test_dir_index.append([current_testing_dir, img_index])

# create val directory and indecies list
num_val_instances = int((args.val_percentage * len(train_dir_index)) / 2)
print 'Extracting val directory and indices from training set({} % ({}) of training images)'.format(args.val_percentage * 100, num_val_instances)
val_dir_index = train_dir_index[
    0:num_val_instances] + train_dir_index[-num_val_instances:]
train_dir_index[0:num_val_instances] = []
train_dir_index[-num_val_instances:] = []

print 'Saving to text files'
testfile = open(args.out_dir + '/test.txt', 'w')
for item in test_dir_index:
    testfile.write('{} {}\n'.format(item[0], item[1]))
testfile.close()

trainfile = open(args.out_dir + '/train.txt', 'w')
for item in train_dir_index:
    trainfile.write('{} {}\n'.format(item[0], item[1]))
trainfile.close()

valfile = open(args.out_dir + '/val.txt', 'w')
for item in val_dir_index:
    valfile.write('{} {}\n'.format(item[0], item[1]))
valfile.close()
print 'Text files saved to {}.'.format(args.out_dir)
