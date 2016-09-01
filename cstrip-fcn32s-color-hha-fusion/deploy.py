#! /usr/bin/python
"""
by Sean McMahon

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
import surgery
import score
from PIL import Image


def fusion_solver(train_net_path, test_net_path, file_location):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 999999999  # do not invoke tests here
    s.test_iter.append(654)
    s.max_iter(3000)
    s.base_lr = 1e-12
    s.lr_policy = 'fixed'
    s.gamma = 0.1
    s.stepsize = 5000
    s.momentum = 9
    s.weight_decay = 0.0005
    s.display(20)
    s.snapshot = 1000
    s.snapshot_prefix = file_location + '/colourHHAsnapshot/fusion_train'
    return s


def fusionNet(hf5_txtfile_path):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label, n.in_data = layers.HDF5Data(
        batch_size=1, source=hf5_txtfile_path, ntop=4)
    n.features_fused = layers.Eltwise(n.color_features, n.hha_features,
                                      operation=params.Eltwise.SUM, coeff=[0.5, 0.5])
    n.upscore = layers.Deconvolution(n.features_fused,
                                     convolution_param=dict(num_output=2, kernel_size=64, stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    # print 'upscore shape {}, in_data shape {}'.format(np.shape(n.upscore), np.shape(n.in_data))
    # print '------------------------------------'
    n.score = crop(n.upscore, n.in_data)
    return n.to_proto()

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer

file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")
# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
from caffe.proto import caffe_pb2
from caffe import layers, params
from caffe.coord_map import crop

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


# Create fusion_test prototxt files
test_net_path = file_location + '/fusion_test.prototxt'
# with open(test_net_path, 'w') as f:
#     f.write(str(fusionNet(os.path.join(file_location, 'testhdf5.txt'))))

# Create and load solver
# solver_path = file_location + '/fusion_solver.prototxt'
# with open(solver_path, 'w') as f:
#     f.write(str(fusion_solver(train_net_path, test_net_path)))
# solver = caffe.SGDSolver(file_location + '/fusion_solver.prototxt')

fusion_fcn = caffe.Net(test_net_path, caffe.TEST)

# Net surgery, filling the deconvolution layer
interp_layers = [k for k in fusion_fcn.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(fusion_fcn, interp_layers)

fusion_fcn.forward()

fusion_im = Image.fromarray(fusion_fcn.blobs['score'].data[
                            0].argmax(0).astype(np.uint8) * 255, mode='P')
img_gt = Image.fromarray(fusion_fcn.blobs['label'].data[
                         0, 0].astype(np.uint8) * 255, mode='P')
# colour_score.save(os.path.join(file_location, 'colourNetOutput.png'))
colourArr = fusion_fcn.blobs['in_data'].data[0].astype(np.uint8)
colourArr = colourArr.transpose((1, 2, 0))  # change to h,w,d
colourArr = colourArr[..., ::-1]  # bgr -> rgb
colourImg = Image.fromarray(colourArr)

overlay = Image.blend(colourImg.convert(
    "RGBA"), fusion_im.convert("RGBA"), 0.7)
overlay.save(os.path.join(file_location, 'fusion_overlay.png'))
overlay_gt = Image.blend(colourImg.convert(
    "RGBA"), img_gt.convert("RGBA"), 0.7)
overlay_gt.save(os.path.join(file_location, 'gt_overlay.png'))
