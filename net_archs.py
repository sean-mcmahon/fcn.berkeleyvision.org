"""
file to house all the architectures being used, could get quite large

By Sean McMahon

"""
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import tempfile
import os.path
# for engine: 1 CAFFE 2 CUDNN
defEngine = 1


def conv_relu(bottom, nout, engineNum=defEngine, ks=3, stride=1, pad=1, lr=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, engine=engineNum,
                         param=[dict(lr_mult=lr, decay_mult=1),
                                dict(lr_mult=lr * 2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True, engine=engineNum)


def max_pool(bottom, engineNum=defEngine, ks=2, engineNumks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride,
                     engine=engineNum)


def mid_fcn_layers(net_spec, convRelu1, engNum, lr_multi, modality=''):
    n = net_spec
    n['conv1_2' + modality], n['relu1_2' + modality] = conv_relu(n[convRelu1],
                                                                 64, engNum,
                                                                 lr=lr_multi)
    n['pool1' + modality] = max_pool(n['relu1_2' + modality], engNum)
    # 2nd set of conv before pool
    n['conv2_1' + modality], n['relu2_1' + modality] = conv_relu(n['pool1' +
                                                                   modality],
                                                                 128, engNum,
                                                                 lr=lr_multi)
    n['conv2_2' + modality], n['relu2_2' + modality] = conv_relu(n['relu2_1' +
                                                                   modality],
                                                                 128, engNum,
                                                                 lr=lr_multi)
    n['pool2' + modality] = max_pool(n['relu2_2' + modality], engNum)
    # 3rd set of conv before pool
    n['conv3_1' + modality], n['relu3_1' + modality] = conv_relu(n['pool2' +
                                                                   modality],
                                                                 256, engNum,
                                                                 lr=lr_multi)
    n['conv3_2' + modality], n['relu3_2' + modality] = conv_relu(n['relu3_1' +
                                                                   modality],
                                                                 256, engNum,
                                                                 lr=lr_multi)
    n['conv3_3' + modality], n['relu3_3' + modality] = conv_relu(n['relu3_2' +
                                                                   modality],
                                                                 256, engNum,
                                                                 lr=lr_multi)
    n['pool3' + modality] = max_pool(n['relu3_3' + modality], engNum)
    # 4th set of conv before pool
    n['conv4_1' + modality], n['relu4_1' + modality] = conv_relu(n['pool3' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['conv4_2' + modality], n['relu4_2' + modality] = conv_relu(n['relu4_1' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['conv4_3' + modality], n['relu4_3' + modality] = conv_relu(n['relu4_2' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['pool4' + modality] = max_pool(n['relu4_3' + modality], engNum)
    # 5th set of conv before pool
    n['conv5_1' + modality], n['relu5_1' + modality] = conv_relu(n['pool4' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['conv5_2' + modality], n['relu5_2' + modality] = conv_relu(n['relu5_1' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['conv5_3' + modality], n['relu5_3' + modality] = conv_relu(n['relu5_2' +
                                                                   modality],
                                                                 512, engNum,
                                                                 lr=lr_multi)
    n['pool5' + modality] = max_pool(n['relu5_3' + modality], engNum)
    return n


def modality_conv_layers(net_spec, data, engNum, lr_multi, modality=''):
    # creates FCN VGG from conv1_1 to pool 5.
    n = net_spec
    # the base net
    convRelu1_n = 'relu1_1' + modality
    n['conv1_1' + modality], n[convRelu1_n] = conv_relu(n[data], 64,
                                                        pad=100)
    return mid_fcn_layers(n, convRelu1_n, engNum, lr_multi, modality=modality)


def fcn(data_split, tops, dropout_prob=0.5, final_multi=1, engineNum=0,
        freeze=False):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='cs_trip_layers',
                               layer='CStripSegDataLayer', ntop=2,
                               param_str=str(dict(
                                   cstrip_dir='/Construction_Site/' +
                                   'Springfield/12Aug16/K2',
                                   split=data_split, tops=tops,
                                   seed=1337)))
    # the base net
    if freeze:
        lr_multi = 0
    else:
        lr_multi = 1
    n.conv1_1, n.relu1_1 = conv_relu(
        n.data, 64, engineNum, pad=100, lr=lr_multi)
    n = mid_fcn_layers(n, 'relu1_1', engineNum, lr_multi)

    # fully conv
    n.fc6, n.relu6 = conv_relu(
        n.pool5, 4096, engineNum,  ks=7, pad=0, lr=lr_multi)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=dropout_prob, in_place=True)
    n.fc7, n.relu7 = conv_relu(
        n.drop6, 4096, engineNum, ks=1, pad=0, lr=lr_multi)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=dropout_prob, in_place=True)

    n.score_fr_trip = L.Convolution(n.drop7, num_output=2, kernel_size=1,
                                    pad=0, engine=engineNum,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=final_multi,
                                                decay_mult=1),
                                           dict(lr_mult=final_multi * 2,
                                                decay_mult=0)])
    n.upscore_trip = L.Deconvolution(n.score_fr_trip,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score = crop(n.upscore_trip, n.data)
    # n.softmax_score = L.Softmax(n.score)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))

    return n


def fcn_early(data_split, tops, dropout_prob=0.5, conv1_1_lr_multi=4,
              final_multi=1, engineNum=0, freeze=False):
    n = caffe.NetSpec()
    if tops[1] != 'depth' and tops[1] != 'hha2' and tops[1] != 'hha':
        raise(
            Exception('Must have hha, hha2, or depth for top[1]; "' +
                      tops + '" tops given'))

    n.color, n[tops[1]],  n.label = L.Python(module='cs_trip_layers',
                                             layer='CStripSegDataLayer', ntop=3,
                                             param_str=str(dict(
                                                 cstrip_dir='/Construction_' +
                                                 'Site/Springfield/12Aug16/K2',
                                                 split=data_split, tops=tops,
                                                 seed=1337)))
    n.data = L.Concat(n.color, n[tops[1]])
    # the base net
    if freeze:
        mid_lr_multi = 0
    else:
        mid_lr_multi = 1
    # n['conv1_1_bgr' + tops[1]], n.relu1_1 = conv_relu(
    #     n.data, 64, engineNum, pad=100, lr=conv1_1_lr_multi)
    n['conv1_1_bgr' + tops[1]] = L.Convolution(n.data, kernel_size=3,
                                               stride=1, num_output=64,
                                               pad=100, engine=engineNum,
                                               weight_filler=dict(
                                                   type='xavier'),
                                               param=[dict(
                                                   lr_mult=conv1_1_lr_multi,
                                                   decay_mult=1),
                                                   dict(
                                                   lr_mult=conv1_1_lr_multi * 2,
                                                   decay_mult=0)])
    n.relu1_1 = L.ReLU(n['conv1_1_bgr' + tops[1]],
                       in_place=True, engine=engineNum)
    n = mid_fcn_layers(n, 'relu1_1', engineNum, mid_lr_multi)

    # fully conv
    n.fc6, n.relu6 = conv_relu(
        n.pool5, 4096, engineNum,  ks=7, pad=0, lr=mid_lr_multi)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=dropout_prob, in_place=True)
    n.fc7, n.relu7 = conv_relu(
        n.drop6, 4096, engineNum, ks=1, pad=0, lr=mid_lr_multi)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=dropout_prob, in_place=True)

    n.score_fr_trip = L.Convolution(n.drop7, num_output=2, kernel_size=1,
                                    pad=0, engine=engineNum,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=final_multi,
                                                decay_mult=1),
                                           dict(lr_mult=final_multi * 2,
                                                decay_mult=0)])
    n.upscore_trip = L.Deconvolution(n.score_fr_trip,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score = crop(n.upscore_trip, n.data)
    # n.softmax_score = L.Softmax(n.score)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))

    return n


def print_rgb_nets():
    tops = ['color', 'label']
    with open('trainval_rgb.prototxt', 'w') as f:
        f.write(str(fcn('train', tops).to_proto()))

    with open('val_rgb.prototxt', 'w') as f:
        f.write(str(fcn('val', tops).to_proto()))

    with open('test_rgb.prototxt', 'w') as f:
        f.write(str(fcn('test', tops).to_proto()))


def print_fcn_early():
    tops = ['color', 'depth',  'label']
    with open('trainval_early.prototxt', 'w') as f:
        f.write(str(fcn_early('train', tops).to_proto()))

    with open('val_early.prototxt', 'w') as f:
        f.write(str(fcn_early('val', tops).to_proto()))

    with open('test_early.prototxt', 'w') as f:
        f.write(str(fcn_early('test', tops).to_proto()))

if __name__ == '__main__':
    print_fcn_early()
