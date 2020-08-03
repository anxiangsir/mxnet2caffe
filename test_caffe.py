import argparse

import caffe
import mxnet as mx
import numpy as np

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument('--mx-model', type=str, default='model_mxnet/mf')
parser.add_argument('--mx-epoch', type=int, default=1)

parser.add_argument('--cf-prototxt', type=str, default='')
parser.add_argument('--cf-model', type=str, default='')
args = parser.parse_args()


def main():
    # input
    INPUT = np.ones((1, 3, 108, 108))

    # caffe
    caffe.set_mode_cpu()
    caffe.set_device(0)
    deploy = args.cf_prototxt
    caffe_model = args.cf_model

    net = caffe.Net(deploy, caffe_model, caffe.TEST)
    net.blobs['data'].data[...] = INPUT
    out = net.forward()
    pre_fc_caffe = net.blobs['fc1'].data
    output_flatten = np.array(pre_fc_caffe, dtype=np.float32)
    print(output_flatten[:, :20])

    # mxnet
    _, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
    all_layers = _.get_internals()
    feature = all_layers['fc1_output']
    model = mx.mod.Module(symbol=feature, data_names=['data'], label_names=[])
    model.bind(data_shapes=[('data', (1, 3, 108, 108))],
               for_training=False, force_rebind=False)
    model.set_params(arg_params, aux_params)
    model.forward(mx.io.DataBatch([mx.nd.array(INPUT)]))
    feature = model.get_outputs()[0].asnumpy()
    print(feature[:, :20])
    print(np.dot(output_flatten, feature.T))


if __name__ == '__main__':
    main()
