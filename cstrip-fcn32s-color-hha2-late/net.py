import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def modality_fcn(net_spec, data, modality):
    n = net_spec
    # the base net
    n['conv1_1' + modality], n['relu1_1' + modality] = conv_relu(n[data], 64,
                                                                 pad=100)
    n['conv1_2' + modality], n['relu1_2' + modality] = conv_relu(n['relu1_1' +
                                                                   modality], 64)
    n['pool1' + modality] = max_pool(n['relu1_2' + modality])

    n['conv2_1' + modality], n['relu2_1' + modality] = conv_relu(n['pool1' +
                                                                   modality], 128)
    n['conv2_2' + modality], n['relu2_2' + modality] = conv_relu(n['relu2_1' +
                                                                   modality], 128)
    n['pool2' + modality] = max_pool(n['relu2_2' + modality])

    n['conv3_1' + modality], n['relu3_1' + modality] = conv_relu(n['pool2' +
                                                                   modality], 256)
    n['conv3_2' + modality], n['relu3_2' + modality] = conv_relu(n['relu3_1' +
                                                                   modality], 256)
    n['conv3_3' + modality], n['relu3_3' + modality] = conv_relu(n['relu3_2' +
                                                                   modality], 256)
    n['pool3' + modality] = max_pool(n['relu3_3' + modality])

    n['conv4_1' + modality], n['relu4_1' + modality] = conv_relu(n['pool3' +
                                                                   modality], 512)
    n['conv4_2' + modality], n['relu4_2' + modality] = conv_relu(n['relu4_1' +
                                                                   modality], 512)
    n['conv4_3' + modality], n['relu4_3' + modality] = conv_relu(n['relu4_2' +
                                                                   modality], 512)
    n['pool4' + modality] = max_pool(n['relu4_3' + modality])

    n['conv5_1' + modality], n['relu5_1' + modality] = conv_relu(n['pool4' +
                                                                   modality], 512)
    n['conv5_2' + modality], n['relu5_2' + modality] = conv_relu(n['relu5_1' +
                                                                   modality], 512)
    n['conv5_3' + modality], n['relu5_3' + modality] = conv_relu(n['relu5_2' +
                                                                   modality], 512)
    n['pool5' + modality] = max_pool(n['relu5_3' + modality])

    # fully conv
    n['fc6' + modality], n['relu6' + modality] = conv_relu(
        n['pool5' + modality], 4096, ks=7, pad=0)
    n['drop6' + modality] = L.Dropout(
        n['relu6' + modality], dropout_ratio=0.5, in_place=True)
    n['fc7' + modality], n['relu7' + modality] = conv_relu(
        n['drop6' + modality], 4096, ks=1, pad=0)
    n['drop7' + modality] = L.Dropout(
        n['relu7' + modality], dropout_ratio=0.5, in_place=True)
    # n['score_fr_trip' + modality] = L.Convolution(
    #     n['drop7' + modality], num_output=2, kernel_size=1, pad=0,
    #     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
    #     weight_filler=dict(type='msra'))
    return n


def fcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.hha2, n.label = L.Python(module='cs_trip_layers',
                                        layer='CStripSegDataLayer', ntop=3,
                                        param_str=str(dict(
                                            cstrip_dir='/Construction_Site/' +
                                            'Springfield/12Aug16/K2', split=split,
                                            tops=tops, seed=1337)))
    n = modality_fcn(n, 'color', 'color')
    n = modality_fcn(n, 'hha2', 'hha2')
    n.score_fused = L.Eltwise(n.score_fr_tripcolor, n.score_fr_triphha2,
                              operation=P.Eltwise.SUM, coeff=[0.5, 0.5])
    n.upscore = L.Deconvolution(n.score_fused,
                                convolution_param=dict(num_output=2,
                                                       kernel_size=64,
                                                       stride=32,
                                                       bias_term=False),
                                param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.color)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))
    return n.to_proto()

def convFusionfcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.hha2, n.label = L.Python(module='cs_trip_layers',
                                        layer='CStripSegDataLayer', ntop=3,
                                        param_str=str(dict(
                                            cstrip_dir='/Construction_Site/' +
                                            'Springfield/12Aug16/K2', split=split,
                                            tops=tops, seed=1337)))
    n = modality_fcn(n, 'color', 'color')
    n = modality_fcn(n, 'hha2', 'hha2')

    n.score_concat = L.Concat(n.fc7color, n.fc7hha2)
    n.conv_fusion1 = L.Convolution(
        n.score_concat, num_output=4096, kernel_size=1, pad=0,
        param=[dict(lr_mult=4, decay_mult=1), dict(lr_mult=8, decay_mult=0)],
        weight_filler=dict(type='msra'))
    n.relu_fusion1 = L.ReLU(n.conv_fusion1, in_place=True)
    n.conv_fusion2 = L.Convolution(
        n.conv_fusion1, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=4, decay_mult=1), dict(lr_mult=8, decay_mult=0)],
        weight_filler=dict(type='msra'))
    n.upscore_fused = L.Deconvolution(n.conv_fusion2,
                                convolution_param=dict(num_output=2,
                                                       kernel_size=64,
                                                       stride=32,
                                                       bias_term=False),
                                param=[dict(lr_mult=0)])
    n.score_fused = crop(n.upscore_fused, n.color)
    n.loss = L.SoftmaxWithLoss(n.score_fused, n.label,
                               loss_param=dict(normalize=False))
    return n.to_proto()

def mixfcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.hha2, n.label = L.Python(module='cs_trip_layers',
                                        layer='CStripSegDataLayer', ntop=3,
                                        param_str=str(dict(
                                            cstrip_dir='/Construction_Site/' +
                                            'Springfield/12Aug16/K2', split=split,
                                            tops=tops, seed=1337)))
    n = modality_fcn(n, 'color', 'color')
    n = modality_fcn(n, 'hha2', 'hha2')
    # find max trip or non trip confidences, cannot use Argmax (no backprop)
    # using eltwise max with split instead
    n.score_colora, n.score_colorb = L.Slice(
        n.score_fr_tripcolor, ntop=2,  slice_param=dict(axis=1))
    n.maxcolor = L.Eltwise(n.score_colora, n.score_colorb,
                           operation=P.Eltwise.MAX)
    n.score_HHA2a, n.score_HHA2b = L.Slice(
        n.score_fr_triphha2, ntop=2,  slice_param=dict(axis=1))
    n.maxhha2 = L.Eltwise(n.score_HHA2a, n.score_HHA2b,
                          operation=P.Eltwise.MAX)
    # concatinate together and softmax for 'probabilites'
    n.maxConcat = L.Concat(n.maxcolor, n.maxhha2, concat_param=dict(axis=1))
    n.maxSoft = L.Softmax(n.maxConcat)
    # separate color and hha using slice layer
    n.probColor, n.probHHA2 = L.Slice(
        n.maxSoft, ntop=2,  slice_param=dict(axis=1))
    # duplicate probabilies using concat layer over dim1 for mulitplication
    n.repProbColor = L.Concat(n.probColor, n.probColor)
    n.repProbHHA2 = L.Concat(n.probHHA2, n.probHHA2)
    # multiply the 'probabilies' with the color and hha scores
    n.weightedColor = L.Eltwise(n.score_fr_tripcolor, n.repProbColor,
                                operation=P.Eltwise.PROD)
    n.weightedHHA2 = L.Eltwise(n.score_fr_triphha2, n.repProbHHA2,
                               operation=P.Eltwise.PROD)
    # combine the prob scores with eltwise summation
    n.score_fused = L.Eltwise(n.weightedColor, n.weightedHHA2,
                              operation=P.Eltwise.SUM, coeff=[1, 1])
    n.upscore = L.Deconvolution(n.score_fused,
                                convolution_param=dict(num_output=2,
                                                       kernel_size=64,
                                                       stride=32,
                                                       bias_term=False),
                                param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.color)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))
    return n.to_proto()


def lateMixfcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.hha2, n.label = L.Python(module='cs_trip_layers',
                                        layer='CStripSegDataLayer', ntop=3,
                                        param_str=str(dict(
                                            cstrip_dir='/Construction_Site/' +
                                            'Springfield/12Aug16/K2', split=split,
                                            tops=tops, seed=1337)))
    n = modality_fcn(n, 'color', 'color')
    n = modality_fcn(n, 'hha2', 'hha2')

    n.upscore_color = L.Deconvolution(n.score_fr_tripNEWcolor,
                                      convolution_param=dict(num_output=2,
                                                             kernel_size=64,
                                                             stride=32,
                                                             bias_term=False),
                                      param=[dict(lr_mult=0)])
    n.score_color = crop(n.upscore_color, n.color)
    n.upscore_hha2 = L.Deconvolution(n.score_fr_tripNEWhha2,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score_hha2 = crop(n.upscore_hha2, n.hha2)

    # find max trip or non trip confidences, cannot use Argmax (no backprop)
    # using eltwise max with split instead
    n.score_colora, n.score_colorb = L.Slice(
        n.score_color, ntop=2,  slice_param=dict(axis=1))
    n.maxcolor = L.Eltwise(n.score_colora, n.score_colorb,
                           operation=P.Eltwise.MAX)
    n.score_HHA2a, n.score_HHA2b = L.Slice(
        n.score_hha2, ntop=2,  slice_param=dict(axis=1))
    n.maxhha2 = L.Eltwise(n.score_HHA2a, n.score_HHA2b,
                          operation=P.Eltwise.MAX)
    # concatinate together and softmax for 'probabilites'
    n.maxConcat = L.Concat(n.maxcolor, n.maxhha2, concat_param=dict(axis=1))
    n.maxSoft = L.Softmax(n.maxConcat)
    # separate color and hha using slice layer
    n.probColor, n.probHHA2 = L.Slice(
        n.maxSoft, ntop=2,  slice_param=dict(axis=1))
    # duplicate probabilies using concat layer over dim1 for mulitplication
    n.repProbColor = L.Concat(n.probColor, n.probColor)
    n.repProbHHA2 = L.Concat(n.probHHA2, n.probHHA2)
    # multiply the 'probabilies' with the color and hha scores
    n.weightedColor = L.Eltwise(n.score_color, n.repProbColor,
                                operation=P.Eltwise.PROD)
    n.weightedHHA2 = L.Eltwise(n.score_hha2, n.repProbHHA2,
                               operation=P.Eltwise.PROD)
    # combine the prob scores with eltwise summation
    n.score_fused = L.Eltwise(n.weightedColor, n.weightedHHA2,
                              operation=P.Eltwise.SUM, coeff=[1, 1])
    n.loss = L.SoftmaxWithLoss(n.score_fused, n.label,
                               loss_param=dict(normalize=False))
    return n.to_proto()


def make_net():
    tops = ['color', 'hha2', 'label']
    # with open('trainval.prototxt', 'w') as f:
    #     f.write(str(fcn('train', tops)))
    # with open('val.prototxt', 'w') as f:
    #     f.write(str(fcn('val', tops)))
    # with open('test.prototxt', 'w') as f:
    #     f.write(str(fcn('test', tops)))
    #
    # with open('trainval_mix.prototxt', 'w') as f:
    #     f.write(str(mixfcn('train', tops)))
    # with open('val_mix.prototxt', 'w') as f:
    #     f.write(str(mixfcn('val', tops)))
    # with open('test_mix.prototxt', 'w') as f:
    #     f.write(str(mixfcn('test', tops)))

    # with open('trainval_latemix2.prototxt', 'w') as f:
    #     f.write(str(lateMixfcn('train', tops)))
    # with open('val_latemix2.prototxt', 'w') as f:
    #     f.write(str(lateMixfcn('val', tops)))
    # with open('test_latemix2.prototxt', 'w') as f:
    #     f.write(str(lateMixfcn('test', tops)))

    with open('trainval_conv.prototxt', 'w') as f:
        f.write(str(convFusionfcn('train', tops)))
    with open('val_conv.prototxt', 'w') as f:
        f.write(str(convFusionfcn('val', tops)))
    with open('test_conv.prototxt', 'w') as f:
        f.write(str(convFusionfcn('test', tops)))
if __name__ == '__main__':
    make_net()
