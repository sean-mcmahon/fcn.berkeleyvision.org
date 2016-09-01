#! /usr/bin/python
"""
by Sean McMahon

"""
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import h5py

file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")
# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='CPU')
args = parser.parse_args()
if 'g' in args.mode or 'G' in args.mode:
    caffe.set_mode_gpu()
    print '-- GPU Mode -- {}'.format(args.mode)
elif 'c' in args.mode or 'C' in args.mode:
    caffe.set_mode_cpu()
    print '-- CPU Mode -- {}'.format(args.mode)
else:
    caffe.set_mode_gpu()
    print '==============='
    print 'No Mode (CPU or GPU) Given'
    print '-- GPU Mode Chosen -- {}'.format(args.mode)
    print '==============='

data_split = 'val'
if data_split == 'train':
    data_split = 'trainval'
val_imgs = np.loadtxt(
    file_parent_dir + '/data/cs-trip/{}.txt'.format(data_split), dtype=str)
save_dir = os.path.join(file_location, data_split + '/')
layer = 'score_fr'
gt = 'label'

color_weights = file_parent_dir + \
    '/cstrip-fcn32s-color/colorSnapshot/_iter_8000.caffemodel'
color_proto = file_parent_dir + \
    '/cstrip-fcn32s-color/{}.prototxt'.format(data_split)
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
score_colour_list = []
input_data_list = []
for counter in range(len(val_imgs)):
    color_net.forward()
    score_colour_list.append(color_net.blobs[layer].data[:])
    input_data_list.append(color_net.blobs['data'].data[:])
    print 'Appending colour features {}/{}'.format(counter, len(val_imgs))
del color_net

hha_weights = file_parent_dir + \
    '/cstrip-fcn32s-hha/HHAsnapshot/train_iter_8000.caffemodel'
hha_proto = file_parent_dir + \
    '/cstrip-fcn32s-hha/{}.prototxt'.format(data_split)
hha_net = caffe.Net(hha_proto, hha_weights, caffe.TEST)
score_hha_list = []
gt_hha_list = []
for counter in range(len(val_imgs)):
    hha_net.forward()
    score_hha_list.append(hha_net.blobs[layer + '_trip'].data[:])
    gt_hha_list.append(hha_net.blobs[gt].data[:])
    print 'Appending hha features {}/{}'.format(counter, len(val_imgs))
del hha_net

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
for counter, idx in enumerate(val_imgs):
    # Write net information to hdf5 file
    val_hdf5_name = os.path.join(save_dir, ''.join(idx) + '.h5')
    with h5py.File(val_hdf5_name, 'w') as f:
        f['color_features'] = score_colour_list[counter]
        f['hha_features'] = score_hha_list[counter]
        f['label'] = gt_hha_list[counter]
        f['in_data'] = input_data_list[counter]
    with open(os.path.join(file_location, 'testhdf5.txt'), 'w') as f:
        f.write(val_hdf5_name + '\n')
