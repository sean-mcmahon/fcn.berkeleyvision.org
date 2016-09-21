#! /usr/bin/python
"""
cstrip deploy_prototxt

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import glob
import score
from PIL import Image


def add_slash(mystring):
    if mystring.endswith('/'):
        return mystring
    else:
        return mystring + '/'

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--iteration', default=8000)
parser.add_argument('--test_type', default='deploy')
parser.add_argument('--network_dir', default='cstrip-fcn32s-color')
parser.add_argument('--snapshot_filter', default='')
parser.add_argument('--save_dir', default=None)
args = parser.parse_args()
iteration = args.iteration
network_dir = args.network_dir
network_dir = add_slash(network_dir)  # ensure slash present
snapshot_filter = args.snapshot_filter
test_type = args.test_type
save_dir = args.save_dir
print 'This is the general deploy_prototxt script!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    working_dir = home_dir + '/Fully-Conv-Network/Resources/FCN_models/' \
        + network_dir
    cstrip_dir = home_dir + '/Construction_Site/Springfield/12Aug16/K2'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    working_dir = home_dir + \
        '/hpc-home/Fully-Conv-Network/Resources/FCN_models/' + network_dir
    cstrip_dir = home_dir + \
        '/hpc-home/Construction_Site/Springfield/12Aug16/K2'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
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
    print '-- GPU Mode Chosen --'
    print '==============='

snapshot_dir = glob.glob(working_dir + '*napshot*')
deploy_prototxt = glob.glob(working_dir + '*' + test_type + '*.prototxt')
weights = snapshot_dir[0] + '/' + snapshot_filter + \
    '_iter_' + str(iteration) + '.caffemodel'
if test_type == 'deploy':
    test_set = np.loadtxt(file_location +
                          '/data/cs-trip/20psdColourImages.txt',
                          dtype=str)
else:
    test_set = np.loadtxt(file_location +
                          '/data/cs-trip/{}.txt'.format(test_type),
                          dtype=str)
if not save_dir:
    save_dir = working_dir + test_type + '_deployImgs'

fcn = caffe.Net(deploy_prototxt[0], weights, caffe.TEST)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
layer = 'score'
for counter, idx in enumerate(test_set):
    fcn.forward()
    im = Image.fromarray(fcn.blobs[layer].data[0].argmax(
        0).astype(np.uint8) * 255, mode='P')
    colorIm = Image.open(
        glob.glob('{}/{}/colour/colourimg_{}_*'.format(cstrip_dir, idx[0],
                                                       idx[1]))[0])
    overlay = Image.blend(colorIm.convert(
        "RGBA"), im.convert("RGBA"), 0.5)
    overlay.save(os.path.join(save_dir, ''.join(idx) + '.png'))
    np.savez_compressed(os.path.join(save_dir, ''.join(idx) + '_layerData'),
                        fcn.blobs[layer].data[0])
    print 'Image data {}/{} saved.'.format(counter, len(test_set))
