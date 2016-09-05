from __future__ import division
import os
import sys
import imp
from os.path import expanduser
import argparse
import glob
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
from caffe import layers, params
from caffe.coord_map import crop
from caffe.proto import caffe_pb2


def writehdf5txt(FCN_Models_dir, outdir, data_splits):
    hdf5txt_locations = []
    if isinstance(data_splits, basestring):  # if data_splits is string
        print 'writehdf5txt: Data splits must be a list or tuple'
        raise()
    for split in data_splits:
        hdf5filenames = glob.glob(os.path.join(
            FCN_Models_dir, 'data/cs-trip/' + split + '_hdf5/*.h5'))
        hdf5txt_locations.append(os.path.join(outdir, split + '_hdf5.txt'))
        txtfile = open(hdf5txt_locations[-1], 'w')
        for filename in hdf5filenames:
            txtfile.write(filename + '\n')
        txtfile.close()
    print 'writehdf5txt: hdf5 file locations written to {}'.format(outdir)
    return hdf5txt_locations


def fixedFusionNet(hf5_txtfile_path, batchSize):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label, n.in_data = layers.HDF5Data(
        batch_size=batchSize, source=hf5_txtfile_path, ntop=4)
    n.features_fused = layers.Eltwise(n.color_features, n.hha_features,
                                      operation=params.Eltwise.SUM,
                                      coeff=[0.5, 0.5])
    n.upscore = layers.Deconvolution(n.features_fused,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    try:
        n.score = crop(n.upscore, n.in_data)
    except:  # because crop cannot handle Concat layers
        print '-------'
        print 'could not initialise crop layer through coord_map.py'
        print '-------'
        n.score = layers.Crop(n.upscore, n.in_data,
                              crop_param=dict(axis=2, offset=19))
    return n.to_proto()


def convFusionNet(hf5_txtfile_path, batchSize):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label, n.in_data = layers.HDF5Data(
        batch_size=batchSize, source=hf5_txtfile_path, ntop=4)
    n.data = layers.Concat(n.color_features, n.hha_features)
    n.score_fr = layers.Convolution(n.data, num_output=2, kernel_size=1, pad=0,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=1, decay_mult=1),
                                           dict(lr_mult=2, decay_mult=0)])
    n.upscore = layers.Deconvolution(n.score_fr,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    try:
        n.score = crop(n.upscore, n.in_data)
        n.loss = layers.SoftmaxWithLoss(n.score, n.label,
                                        loss_param=dict(normalize=False))
    except:  # because crop cannot handle Concat layers
        print '-------'
        print 'could not initialise crop layer through coord_map.py'
        print '-------'
        n.score = layers.Crop(n.upscore, n.in_data,
                              crop_param=dict(axis=2, offset=19))
        n.loss = layers.SoftmaxWithLoss(
            n.score, n.label, loss_param=dict(normalize=False))

    return n.to_proto()


def createNets(split='test'):
    [hdf5files] = writehdf5txt(file_parent_dir, file_location, [split])
    batch_size = 1
    conv_test_net_path = file_location + '/fusionConv_' + split + '.prototxt'
    with open(conv_test_net_path, 'w') as f:
        f.write(str(convFusionNet(hdf5files, batch_size)))

    batch_size = 1
    fixed_test_net_path = os.path.join(
        file_location, 'fusionFixed_' + split + '.prototxt')
    with open(fixed_test_net_path, 'w') as f:
        f.write(str(fixedFusionNet(hdf5files, batch_size)))

    return [fixed_test_net_path, conv_test_net_path]

if __name__ == '__main__':
    # User Input
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--net', default='convFusionNet')
    # args = parser.parse_args()
    [fixed_net_path, conv_net_path] = createNets()
    print 'prototxt files saved as \n- {}\n- {}'.format(fixed_net_path,
                                                        conv_net_path)
