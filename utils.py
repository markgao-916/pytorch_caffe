# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L, params as P
import math
import numpy as np

ACTIVATION_FN_DICT = {'relu': L.ReLU}
num_class=6



def deconv(netspec, input_node, num_output, scope, ks=4, stride=2, pad=1, std=0.01):
    """
    :param input_node:
    :param num_output:
    :param ks:
    :param stride:
    :param pad:
    :param std:
    :return:
    """
    n = netspec
    n.node = L.Deconvolution(input_node,convolution_param=dict(num_output=num_output,
                                    kernel_size=ks,
                                    stride=stride,
                                    pad=pad,
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0)),
                                    param=[dict(lr_mult=0)],
                                    name='%s' % scope)
    return n.node


def conv2d(netspec, input_node, num_output, kernel_size, stride,
           activation_fn='relu',
           bias_term=False,
           use_bn=True,
           scope=None,
           padding='SAME'):
    """

    :param netspec:
    :param input_node:
    :param num_output: int
    :param kernel_size: int
    :param stride: int
    :param activation_fn:
    :param bn: bool
    :param scope: string or None
    :param prune_rate: float, keep rate for prune
    :return:
    """
    n = netspec
    if padding == 'SAME':
        pad = kernel_size // 2
    elif padding == 'VALID':
        pad = 0
    # - - - - - - - - - - - - - - - - - - - -
    # In slim, if `normalizer_fn` is provided then `biases_initializer` and
    # `biases_regularizer` are ignored and `biases` are not created nor added.
    # default set to None for no normalizer function
    bias_term = ((not use_bn) and bias_term)
    # - - - - - - - - - - - - - - - - - - - -

    n.node = L.Convolution(input_node, kernel_size=kernel_size, stride=stride, pad=pad,
                           num_output=num_output,
                           bias_term=bias_term,
                           name='%s' % scope)

    if use_bn:
        n.node = L.BatchNorm(n.node,
                             batch_norm_param=dict(use_global_stats=True,
                                                   moving_average_fraction=0.9,
                                                   eps=1e-5),
                             in_place=True,
                             name='%s/BatchNorm' % scope)
        n.node = L.Scale(n.node, in_place=True, scale_param=dict(bias_term=True),
                         name='%s/scale' % scope)
    if activation_fn is not None:
        # act_fn = ACTIVATION_FN_DICT[activation_fn]
        # n.node = act_fn(n.node, in_place=False)
        n.node = L.ReLU(n.node, in_place=True, name='%s/relu' % scope)

    return n.node


def depthwise_conv2d(netspec, input_node, num_output, kernel_size,
                     stride,
                     activation_fn='relu',
                     use_bn=True,
                     scope=None,
                     bias_term=False):
    """

    :param netspec:
    :param input_node:
    :param num_output: int, num_output should be equal to num_channels of input_node
    :param kernel_size: int
    :param stride: int
    :param activation_fn: string
    :param use_bn: bool
    :param scope:
    :return:
    """
    n = netspec
    pad = math.floor(kernel_size / 2)
    n.node = L.Convolution(input_node, kernel_size=kernel_size, stride=stride, pad=pad,
                           num_output=num_output,
                           group=num_output,
                           bias_term=bias_term,
                           name=scope)

    if use_bn:
        n.node = L.BatchNorm(n.node,
                             batch_norm_param=dict(use_global_stats=True,
                                                   moving_average_fraction=0.9,
                                                   eps=1e-5,),
                             in_place=True,
                             name='%s/BatchNorm' % scope)
        n.node = L.Scale(n.node, in_place=True, scale_param=dict(bias_term=True),
                         name='%s/scale' % scope)
    if activation_fn is not None:
        # act_fn = ACTIVATION_FN_DICT[activation_fn]
        # n.node = act_fn(n.node, in_place=False)
        n.node = L.ReLU(n.node, in_place=True, name='%s/relu' % scope)

    return n.node


def bottleneck(netspec, input_node, c_e, c_o, stride, scope, resnet_block=True):
    """
    conv_expand_1x1 -> dwise_conv_3x3 -> conv_linear_1x1 -> add_input_or_not

    :param netspec:
    :param input_node:
    :param c_e: int, num_output of expand layer
    :param c_o: int, num_output of bottleneck
    :param stride: int, stride of bottleneck
    :param scope:
    :param resnet_block: bool, add input_node to result or not
    :return:
    """
    n = netspec

    # expand
    n.node = conv2d(netspec, input_node, num_output=c_e,
                    kernel_size=1, stride=1,
                    activation_fn='relu',
                    use_bn=True,
                    padding='VALID',
                    scope='%s/conv1' % scope)
    # dwise
    n.node = depthwise_conv2d(netspec, n.node, num_output=c_e,
                              kernel_size=3,
                              stride=stride,
                              activation_fn='relu',
                              scope='%s/conv2' % scope)
    # linear
    n.node = conv2d(netspec, n.node, num_output=c_o,
                    kernel_size=1, stride=1,
                    activation_fn=None,
                    use_bn=True,
                    padding='VALID',
                    scope='%s/conv3' % scope)
    if resnet_block:
        n.block_o = L.Eltwise(input_node, n.node, eltwise_param=dict(operation=1), name='%s_add' % scope)
    else:
        n.block_o = n.node
    return n.block_o

def heartmap(netspec, input_node,c_i, c_o, scope):
    """
    dwise_conv_3x3 -> conv_linear_1x1 -> conv_linear_1x1

    :param netspec:
    :param input_node:
    :param c_o: int, num_output of bottleneck
    :param scope:
    :return:
    """
    n = netspec
    # dwise
    # c_i=input_node
    n.node = depthwise_conv2d(netspec, input_node, num_output=c_i,
                              kernel_size=3,
                              stride=1,
                              activation_fn=None,
                              scope='%s/0' % scope,
                              bias_term=True)
    # linear
    n.node = conv2d(netspec, n.node, num_output=64,
                    kernel_size=1, stride=1,
                    activation_fn='relu',
                    use_bn=False,
                    scope='%s/2' % scope)
    n.node = conv2d(netspec, n.node, num_output=c_o,
                    kernel_size=1, stride=1,
                    activation_fn=None,
                    use_bn=False,
                    bias_term=True,
                    scope='%s/4' % scope)
    return n.node


def ori_heartmap(netspec, input_node,c_i, c_o, scope):
    """
    dwise_conv_3x3 -> conv_linear_1x1 -> conv_linear_1x1

    :param netspec:
    :param input_node:
    :param c_o: int, num_output of bottleneck
    :param scope:
    :return:
    """
    n = netspec
    # linear
    n.node = conv2d(netspec, input_node, num_output=c_i,
                    kernel_size=3, stride=1,
                    activation_fn='relu',
                    use_bn=False,
                    bias_term=True,
                    scope='%s/0' % scope)
    n.node = conv2d(netspec, n.node, num_output=c_o,
                    kernel_size=1, stride=1,
                    activation_fn=None,
                    use_bn=False,
                    bias_term=True,
                    scope='%s/2' % scope)
    return n.node


def subnet(netspec, input_node, c_i, scope):
    n = netspec
    n.wh = ori_heartmap(netspec, input_node, c_i, 2, scope+'/wh')
    n.hm = ori_heartmap(netspec, input_node, c_i, num_class, scope+'/hm')
    n.reg = ori_heartmap(netspec, input_node, c_i, 2, scope+'/reg')
    return [n.hm, n.wh, n.reg]


def get_npy(model_path, npy_file):
    """
    save model weights as .npy
    :param model_path:
    :param npy_file:
    :return:
    """
    pass
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         saver = tf.train.import_meta_graph(meta_file)
    #         saver.restore(sess, ckpt_file)
    #
    #         av = tf.trainable_variables()
    #         data_dict = dict()
    #         for var in av:
    #             var_name = var.name.split(':')[0]
    #             if var_name == 'Variable':
    #                 continue
    #             val = sess.run(var)
    #             data_dict.update({var_name: val})
    #             print('Got %s, shape: ' % var_name, val.shape)
    #
    #         np.save(npy_file, data_dict)


def decode_npy_model(npy_file):
    data_dict = np.load(npy_file, allow_pickle=True).item()
    return data_dict

def decode_pth_model(pth_path):
    import torch
    checkpoint = torch.load(pth_path)
    data_dict = checkpoint['state_dict']
    keys = data_dict.keys()
    npy_dict = {}
    for key in keys:
        val = data_dict[key].cpu().numpy()
        npy_dict[key]=val
    return npy_dict
