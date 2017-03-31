import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


defEngine = 1


def conv_relu(bottom, nout, engineNum=defEngine, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, engine=engineNum,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True, engine=engineNum)


def max_pool(bottom, engineNum=defEngine, , engineNumks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride,
                     engine=engineNum)


def fcn_rgb(split, tops, engineNum=1):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='cs_trip_layers',
                               layer='CStripSegDataLayer', ntop=2,
                               param_str=str(dict(
                                   cstrip_dir='/Construction_Site/' +
                                   'Springfield/12Aug16/K2',
                                   split=split, tops=tops,
                                   seed=1337)))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, engineNum, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, engineNum)
    n.pool1 = max_pool(n.relu1_2, engineNum)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, engineNum)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, engineNum)
    n.pool2 = max_pool(n.relu2_2, engineNum)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, engineNum)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, engineNum)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, engineNum)
    n.pool3 = max_pool(n.relu3_3, engineNum)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, engineNum)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, engineNum)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, engineNum)
    n.pool4 = max_pool(n.relu4_3, engineNum)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, engineNum)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, engineNum)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, engineNum)
    n.pool5 = max_pool(n.relu5_3, engineNum)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, engineNum,  ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, engineNum, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score_fr_trip = L.Convolution(n.drop7, num_output=2, kernel_size=1,
                                    pad=0, engine=engineNum,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=5, decay_mult=1),
                                           dict(lr_mult=10, decay_mult=0)])
    n.upscore_trip = L.Deconvolution(n.score_fr_trip,
                                     convolution_param=dict(num_output=2,
                                                            kernel_size=64,
                                                            stride=32,
                                                            bias_term=False),
                                     param=[dict(lr_mult=0)])
    n.score = crop(n.upscore_trip, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False))

    return n


def createNet(net_type='rgb', split, tops):
    engineNum = 1  # 1 CAFFE 2 CUDNN
    if net_type == 'rgb' or net_type == 'RGB':
        net = fcn_rgb(split, tops, engineNum)
    else:
        Exception('net_type {}, unrecognised create case for new network here.')
    return net


def print_net():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn_rgb('train', tops).to_proto()))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn_rgb('val', tops).to_proto()))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn_rgb('test', tops).to_proto()))

if __name__ == '__main__':
    print_net()
