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
parser.add_argument('--training_dir', nargs='+', help='List of directories', default='/home/sean/hpc-home/Construction_Site/Springfield/12Aug16/K2/2016-08-12-10-36-59_4thFloorApartments')
parser.add_argument('--testing_dir', default='')
parser.add_argument('--val_percentage', default=0.15)
args = parser.parse_args()

train_image_names = []; test_image_names = []
for train_path in args.training_dir:
    directory = getSubSetName(train_path)
    img_fullnames = glob.glob(train_path+'/colour/*.png')
    for image_name in img_fullnames:
        img_index = getImgIndex(os.path.basename(image_name))
        train_image_names.append([directory, img_index])
        print 'Index {}\nDirectory {}'.format(img_index,directory)
        # this is creating a list of lists!

    print 'looping, train_path={}'.format(train_path)
for test_path in args.testing_dir:
    test_image_names.append(glob.glob(test_path+'/colour/*.png'))

print 'train list has {} elements, sub-element length {}\nExample sub-sub-element: {}'.format(len(train_image_names),len(train_image_names[-1]), train_image_names[-1][-1])
