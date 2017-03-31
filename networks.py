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


def fcn_rgb(split, tops, dropout_prob=0.5, final_multi=1, engineNum=0, freeze=False):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='cs_trip_layers',
                               layer='CStripSegDataLayer', ntop=2,
                               param_str=str(dict(
                                   cstrip_dir='/Construction_Site/' +
                                   'Springfield/12Aug16/K2',
                                   split=split, tops=tops,
                                   seed=1337)))

    # the base net
    if freeze:
        lr_multi = 0
    else:
        lr_multi = 1
    n.conv1_1, n.relu1_1 = conv_relu(
        n.data, 64, engineNum, pad=100, lr=lr_multi)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, engineNum, lr=lr_multi)
    n.pool1 = max_pool(n.relu1_2, engineNum)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, engineNum, lr=lr_multi)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, engineNum, lr=lr_multi)
    n.pool2 = max_pool(n.relu2_2, engineNum)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, engineNum, lr=lr_multi)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, engineNum, lr=lr_multi)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, engineNum, lr=lr_multi)
    n.pool3 = max_pool(n.relu3_3, engineNum)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, engineNum, lr=lr_multi)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, engineNum, lr=lr_multi)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, engineNum, lr=lr_multi)
    n.pool4 = max_pool(n.relu4_3, engineNum)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, engineNum, lr=lr_multi)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, engineNum, lr=lr_multi)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, engineNum, lr=lr_multi)
    n.pool5 = max_pool(n.relu5_3, engineNum)

    # fully conv
    n.fc6, n.relu6 = conv_relu(
        n.pool5, 4096, engineNum,  ks=7, pad=0, lr=lr_multi)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=dropout_prob, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, engineNum, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=dropout_prob, in_place=True)

    n.score_fr_trip = L.Convolution(n.drop7, num_output=2, kernel_size=1,
                                    pad=0, engine=engineNum,
                                    weight_filler=dict(type='xavier'),
                                    param=[dict(lr_mult=final_multi, decay_mult=1),
                                           dict(lr_mult=final_multi * 2, decay_mult=0)])
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


def createNet(split, net_type='rgb', f_multi=5, dropout_prob=0.5, engine=1):

    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        net = fcn_rgb(split, tops, engineNum=engine, final_multi=f_multi,
                      dropout_prob=dropout_prob)
    else:
        Exception('net_type {}, unrecognised create case for new network here.')
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(net.to_proto()))
        return f.name


def print_net(path, split='test', net_type='rgb'):
    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        with open(os.path.join(path, split + '.prototxt'), 'w') as f:
            f.write(str(fcn_rgb(split, tops).to_proto()))
    else:
        Exception('net_type {}, unrecognised create case for new network here.')


def print_all_nets():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn_rgb('train', tops).to_proto()))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn_rgb('val', tops).to_proto()))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn_rgb('test', tops).to_proto()))

if __name__ == '__main__':
    print_all_nets()
