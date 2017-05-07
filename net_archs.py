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
    n['conv1_1' + modality], n[convRelu1_n] = conv_relu(n[data], 64, engNum,
                                                        lr=lr_multi,
                                                        pad=100)
    return mid_fcn_layers(n, convRelu1_n, engNum, lr_multi, modality=modality)


def modality_fcn(net_spec, data, modality, engNum, lr_multi, dropout_prob,
                 final_multi, new_final_fc=False):
    n = net_spec
    n = modality_conv_layers(n, data, engNum, lr_multi, modality)
    # fully conv
    n['fc6' + modality], n['relu6' + modality] = conv_relu(
        n['pool5' + modality], 4096, engNum, ks=7, pad=0, lr=lr_multi)
    n['drop6' + modality] = L.Dropout(
        n['relu6' + modality], dropout_ratio=dropout_prob, in_place=True)
    n['fc7' + modality], n['relu7' + modality] = conv_relu(
        n['drop6' + modality], 4096, engNum, ks=1, pad=0, lr=lr_multi)
    n['drop7' + modality] = L.Dropout(
        n['relu7' + modality], dropout_ratio=dropout_prob, in_place=True)
    if new_final_fc:
        n['score_fr_trip' + modality + '_new'] = L.Convolution(
            n['drop7' + modality], num_output=2, kernel_size=1, pad=0,
            param=[dict(lr_mult=final_multi, decay_mult=1),
                   dict(lr_mult=2 * final_multi, decay_mult=0)],
            weight_filler=dict(type='msra'))
    else:
        n['score_fr_trip' + modality] = L.Convolution(
            n['drop7' + modality], num_output=2, kernel_size=1, pad=0,
            param=[dict(lr_mult=final_multi, decay_mult=1),
                   dict(lr_mult=2 * final_multi, decay_mult=0)],
            weight_filler=dict(type='msra'))
    return n


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


def fcn_conv(data_split, tops, dropout_prob=0.5,
             final_multi=1, engineNum=0, freeze=True):
    n = caffe.NetSpec()
    if tops[1] != 'depth' and tops[1] != 'hha2' and tops[1] != 'hha':
        raise(
            Exception('Must have hha, hha2, or depth for top[1]; "' +
                      tops + '" tops given'))
    if freeze:
        conv_multi = 0
    else:
        conv_multi = 1

    n.color, n[tops[1]],  n.label = L.Python(module='cs_trip_layers',
                                             layer='CStripSegDataLayer', ntop=3,
                                             param_str=str(dict(
                                                 cstrip_dir='/Construction_' +
                                                 'Site/Springfield/12Aug16/K2',
                                                 split=data_split, tops=tops,
                                                 seed=1337)))
    n = modality_conv_layers(n, 'color', engineNum,
                             conv_multi, modality='color')
    n = modality_conv_layers(n, tops[1], engineNum,
                             conv_multi, modality=tops[1])

    for modal in ['color', tops[1]]:
        n['fc6' + modal], n['relu6' + modal] = conv_relu(n['pool5' + modal],
                                                         4096, ks=7, pad=0)
        n['drop6' + modal] = L.Dropout(n['relu6' + modal],
                                       dropout_ratio=0.5, in_place=True)

    # Let the conv fusion begin! Mwahahaha!!
    # I want these layers to be randomly initialised, as pre-trained colour
    # features tend to make the network only learn colour features
    n.fc6_concat = L.Concat(n['drop6color'], n['drop6' + tops[1]])
    n.fc7fuse = L.Convolution(n.fc6_concat, kernel_size=1, stride=1,
                              num_output=4096, pad=0, engine=engineNum,
                              param=[dict(lr_mult=final_multi, decay_mult=1),
                                     dict(lr_mult=final_multi * 2, decay_mult=0)],
                              weight_filler=dict(type='msra'))
    n.relu7fuse = L.ReLU(n.fc7fuse, in_place=True, engine=engineNum)
    n.drop7fuse = L.Dropout(
        n.relu7fuse, dropout_ratio=dropout_prob, in_place=True)
    n.score_fr_tripfuse = L.Convolution(n.drop7fuse, num_output=2,
                                        kernel_size=1, pad=0,
                                        param=[dict(lr_mult=final_multi,
                                                    decay_mult=1),
                                               dict(lr_mult=final_multi * 2,
                                                    decay_mult=0)],
                                        weight_filler=dict(type='msra'))
    # Upscale scores
    n.upscore_trip = L.Deconvolution(n.score_fr_tripfuse,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score = crop(n.upscore_trip, n.color)
    # n.softmax_score = L.Softmax(n.score)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))
    return n

# test code!


def mixfcn(data_split, tops, dropout_prob=0.5,
           final_multi=1, engineNum=0, freeze=True):
            #  split, tops):
    n = caffe.NetSpec()
    if tops[1] != 'depth' and tops[1] != 'hha2' and tops[1] != 'hha':
        raise(
            Exception('Must have hha, hha2, or depth for top[1]; "' +
                      tops + '" tops given'))
    if freeze:
        base_lr_multi = 0
    else:
        base_lr_multi = 1

    n.color, n[tops[1]],  n.label = L.Python(module='cs_trip_layers',
                                             layer='CStripSegDataLayer', ntop=3,
                                             param_str=str(dict(
                                                 cstrip_dir='/Construction_' +
                                                 'Site/Springfield/12Aug16/K2',
                                                 split=data_split, tops=tops,
                                                 seed=1337)))
# modality_fcn(net_spec, data, modality, engNum, lr_multi, dropout_prob,
    #  final_multi, new_final_fc)
    n = modality_fcn(n, 'color', 'color', engineNum, base_lr_multi,
                     dropout_prob, final_multi, new_final_fc=True)
    n = modality_fcn(n, tops[1], tops[1], engineNum, base_lr_multi,
                     dropout_prob, final_multi, new_final_fc=True)

    # Upscale modality predictions
    n.upscorecolor = L.Deconvolution(n['score_fr_tripcolor_new'],
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score_color = crop(n.upscorecolor, n.color)
    n['upscore' + tops[1]] = L.Deconvolution(n['score_fr_trip' + tops[1] + '_new'],
                                             convolution_param=dict(num_output=2,
                                                                    kernel_size=64,
                                                                    stride=32,
                                                                    bias_term=False),
                                             param=[dict(lr_mult=0)])
    n['score_' + tops[1]] = crop(n['upscore' + tops[1]], n[tops[1]])

    # find max trip or non trip confidences, cannot use Argmax (no backprop)
    # using eltwise max with split instead
    n.score_colora, n.score_colorb = L.Slice(n.score_color,
                                             ntop=2,  slice_param=dict(axis=1))
    n.maxcolor = L.Eltwise(n.score_colora, n.score_colorb,
                           operation=P.Eltwise.MAX)
    n['score_' + tops[1] + 'a'], n['score_' + tops[1] + 'b'] = L.Slice(
        n['score_' + tops[1]], ntop=2,  slice_param=dict(axis=1))
    n['max' + tops[1]] = L.Eltwise(n['score_' + tops[1] + 'a'],
                                   n['score_' + tops[1] + 'b'],
                                   operation=P.Eltwise.MAX)
    # concatinate together and softmax for 'probabilites'
    n.maxConcat = L.Concat(n.maxcolor, n['max' + tops[1]],
                           concat_param=dict(axis=1))
    n.maxSoft = L.Softmax(n.maxConcat)
    # separate color and hha using slice layer
    n.probColor, n['prob' + tops[1]] = L.Slice(
        n.maxSoft, ntop=2,  slice_param=dict(axis=1))
    # duplicate probabilies using concat layer over dim1 for mulitplication
    n.repProbColor = L.Concat(n.probColor, n.probColor)
    n['repProb' + tops[1]] = L.Concat(n['prob' + tops[1]], n['prob' + tops[1]])
    # multiply the 'probabilies' with the color and hha scores
    n.weightedColor = L.Eltwise(n.score_color, n.repProbColor,
                                operation=P.Eltwise.PROD)
    n['weighted' + tops[1]] = L.Eltwise(n['score_' + tops[1]],
                                        n['repProb' + tops[1]],
                                        operation=P.Eltwise.PROD)
    # combine the prob scores with eltwise summation
    n.score_fused = L.Eltwise(n.weightedColor, n['weighted' + tops[1]],
                              operation=P.Eltwise.SUM, coeff=[1, 1])

    n.loss = L.SoftmaxWithLoss(n.score_fused, n.label,
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


def print_fcn_conv():
    tops = ['color', 'depth',  'label']
    with open('trainval_conv.prototxt', 'w') as f:
        f.write(str(fcn_conv('train', tops).to_proto()))

    with open('val_conv.prototxt', 'w') as f:
        f.write(str(fcn_conv('val', tops).to_proto()))

    with open('test_conv.prototxt', 'w') as f:
        f.write(str(fcn_conv('test', tops).to_proto()))


def print_mixfcn():
    tops = ['color', 'hha2',  'label']
    with open('trainval_mix.prototxt', 'w') as f:
        f.write(str(mixfcn('train', tops).to_proto()))

    with open('val_mix.prototxt', 'w') as f:
        f.write(str(mixfcn('val', tops).to_proto()))

    with open('test_mix.prototxt', 'w') as f:
        f.write(str(mixfcn('test', tops).to_proto()))
if __name__ == '__main__':
    print_mixfcn()
