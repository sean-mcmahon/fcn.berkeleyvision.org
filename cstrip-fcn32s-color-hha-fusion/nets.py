import os
import sys
from os.path import expanduser
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
from caffe.proto import caffe_pb2
from caffe import layers, params
from caffe.coord_map import crop


def fixedFusionNet(hf5_txtfile_path, batchSize):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label, n.in_data = layers.HDF5Data(
        batch_size=batchSize, source=hf5_txtfile_path, ntop=4)
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


def convFusionNet(hf5_txtfile_path, batchSize):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label, n.in_data = layers.HDF5Data(
        batch_size=batchSize, source=hf5_txtfile_path, ntop=4)
    n.data = layers.Concat(n.color_features, n.hha_features)
    n.score_fr = layers.Convolution(n.data, num_output=2, kernel_size=1, pad=0,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = layers.Deconvolution(n.features_fused,
                                     convolution_param=dict(num_output=2, kernel_size=64, stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.in_data)
    n.loss = layers.SoftmaxWithLoss(n.score, n.label,
                                    loss_param=dict(normalize=False))
    return n.to_proto()
