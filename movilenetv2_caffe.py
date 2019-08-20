# # -*- coding: utf-8 -*-
# import caffe
# import caffe.proto.caffe_pb2 as caffe_pb2
# from caffe import layers as L, params as P
# import numpy as np
# from utils import conv2d, depthwise_conv2d, bottleneck,heartmap,subnet,deconv_relu
#
# from utils import get_npy, decode_npy_model, decode_pth_model
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# def mobilenetv2_centernet_inference(netspec, input_node):
#     layer_cfg = [['conv2d', 32, 3, 2, 'relu', False, True, 'conv1'],
#                   ['bottleneck_0_0', 32, 16, 1, 'LinearBottleneck0_0'],
#                   ['bottleneck_1_0', 96,24, 2, 'LinearBottleneck1_0'],
#                   ['bottleneck_1_1', 144,24, 1, 'LinearBottleneck1_1'],
#                   ['bottleneck_2_0', 144,32, 2, 'LinearBottleneck2_0'],
#                   ['bottleneck_2_1', 192,32, 1, 'LinearBottleneck2_1'],
#                   ['bottleneck_3_0', 192,64, 2, 'LinearBottleneck3_0'],
#                   ['bottleneck_3_1', 384,64, 1, 'LinearBottleneck3_1'],
#                   ['bottleneck_3_2', 384,64, 1, 'LinearBottleneck3_2'],
#                   ['bottleneck_4_0', 384,96, 1, 'LinearBottleneck4_0'],
#                   ['bottleneck_4_1', 576,96, 1, 'LinearBottleneck4_1'],
#                   ['bottleneck_4_2', 576,96, 1, 'LinearBottleneck4_2'],
#                   ['bottleneck_5_0', 576,160, 2, 'LinearBottleneck5_0'],
#                   ['bottleneck_5_1', 960,160, 1, 'LinearBottleneck5_1'],
#                   ['bottleneck_6_0', 960,320, 1, 'LinearBottleneck6_0']]
#     heartmap_cfg=[['heartmap', 32, 'MapHeatmap_2'],
#                   ['heartmap', 96, 'MapHeatmap_4'],
#                   ['heartmap', 320, 'MapHeatmap_6']]
#
#     n = netspec
#     blobs_lst = []
#     layer = layer_cfg[0]
#     n.conv = conv2d(n, input_node, num_output=32, kernel_size=3,
#                     stride=2,
#                     activation_fn='relu',
#                     bias_term=False,
#                     use_bn=True,
#                     scope='conv1')
#     resnet_block = False
#     layer = layer_cfg[1]
#     n.bottleneck_0_0 = bottleneck(n, n.conv, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[2]
#     n.bottleneck_1_0 = bottleneck(n, n.bottleneck_0_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[3]
#     n.bottleneck_1_1 = bottleneck(n, n.bottleneck_1_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[4]
#     n.bottleneck_2_0 = bottleneck(n, n.bottleneck_1_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[5]
#     n.bottleneck_2_1 = bottleneck(n, n.bottleneck_2_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[6]
#     n.bottleneck_3_0 = bottleneck(n, n.bottleneck_2_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[7]
#     n.bottleneck_3_1 = bottleneck(n, n.bottleneck_3_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[8]
#     n.bottleneck_3_2 = bottleneck(n, n.bottleneck_3_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[9]
#     n.bottleneck_4_0 = bottleneck(n, n.bottleneck_3_2, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[10]
#     n.bottleneck_4_1 = bottleneck(n, n.bottleneck_4_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[11]
#     n.bottleneck_4_2 = bottleneck(n, n.bottleneck_4_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[12]
#     n.bottleneck_5_0 = bottleneck(n, n.bottleneck_4_2, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[13]
#     n.bottleneck_5_1 = bottleneck(n, n.bottleneck_5_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#     layer = layer_cfg[14]
#     n.bottleneck_6_0 = bottleneck(n, n.bottleneck_5_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
#                                   resnet_block=resnet_block)
#
#     n.deconv1 = L.de
#     heartmap_layer = heartmap_cfg[0]
#     n.hm1, n.wh1, n.reg1 = subnet(n, n.bottleneck_2_1, c_i=heartmap_layer[1], scope=heartmap_layer[2])
#     heartmap_layer = heartmap_cfg[1]
#     n.hm2, n.wh2, n.reg2 = subnet(n, n.bottleneck_4_2, c_i=heartmap_layer[1], scope=heartmap_layer[2])
#     heartmap_layer = heartmap_cfg[2]
#     n.hm3, n.wh3, n.reg3 = subnet(n, n.bottleneck_6_0, c_i=heartmap_layer[1], scope=heartmap_layer[2])
#
#     return n
#
#
# def create_model(depth_coe=1.):
#     n = caffe.NetSpec()
#
#     n.data = L.Input(shape=[dict(dim=[1, 3, 512, 512])], ntop=1)
#     n = mobilenetv2_centernet_inference(n, n.data)
#
#     return n
#
#
#
# def parse_caffemodel(caffemodel):
#     MODEL_FILE = '/home/amax/workspace/pytorch_caffe/deploy.prototxt'
#     # 预先训练好的caffe模型
#     PRETRAIN_FILE = caffemodel
#
#     # 保存参数的文件
#     params_txt = 'params.txt'
#     pf = open(params_txt, 'w')
#
#     # 让caffe以测试模式读取网络参数
#     net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
#
#     # 遍历每一层
#     for param_name in net.params.keys():
#         # 权重参数
#         weight = net.params[param_name][0].data
#         # 偏置参数
#         bias = net.params[param_name][1].data
#
#         # 该层在prototxt文件中对应“top”的名称
#         pf.write(param_name)
#         pf.write('\n')
#
#         # 写权重参数
#         pf.write('\n' + param_name + '_weight:\n\n')
#         # 权重参数是多维数组，为了方便输出，转为单列数组
#         weight.shape = (-1, 1)
#
#         for w in weight:
#             pf.write('%ff, ' % w)
#
#         # 写偏置参数
#         pf.write('\n\n' + param_name + '_bias:\n\n')
#         # 偏置参数是多维数组，为了方便输出，转为单列数组
#         bias.shape = (-1, 1)
#         for b in bias:
#             pf.write('%ff, ' % b)
#
#         pf.write('\n\n')
#
#     pf.close()
#
#     print('--')
#
# def gen_prototxt(model_name='MobileNet_CenterNet'):
#     net = create_model()
#     with open('%s.prototxt' % model_name, 'w') as f:
#         f.write(str(net.to_proto()))
#
# def save_conv2caffe(weights=None, biases=None, conv_param=None):
#     if conv_param is not None:
#         if biases is not None:
#             conv_param[1].data[...] = biases
#         if weights is not None:
#             conv_param[0].data[...] = weights
#
#
# def save_fc2caffe(weights, biases, fc_param):
#     print(biases.size(), weights.size())
#     print(fc_param[1].data.shape)
#     print(fc_param[0].data.shape)
#     fc_param[1].data[...] = biases
#     fc_param[0].data[...] = weights
#
#
# def save_bn2caffe(running_mean=None, running_var=None, bn_param=None):
#     if bn_param is not None:
#         if running_mean is not None:
#             bn_param[0].data[...] = running_mean
#         if running_var is not None:
#             bn_param[1].data[...] = running_var
#         bn_param[2].data[...] = np.array([1.0])
#
#
# def save_scale2caffe(weights=None, biases=None, scale_param=None):
#     if scale_param is not None:
#         if biases is not None:
#             scale_param[1].data[...] = biases
#         if weights is not None:
#             scale_param[0].data[...] = weights
# def map_torch_bn_layer_to_caffe_bn(bn_layer_name):
#     layer_name = bn_layer_name.replace('bn', 'conv')
#     lst = layer_name.split('.')
#     if 'run' in layer_name:
#         new_lst = lst[2:-1]+['BatchNorm']
#     else:
#         new_lst = lst[2:-1] + ['scale']
#     caffe_bn_layer_name = '/'.join(new_lst)
#     return caffe_bn_layer_name
#
#
#
# def save_caffemodel(model_name,pth_path=None):
#     # if meta_file is not None and ckpt_file is not None:
#     #     convert_meta_to_npy(meta_file, ckpt_file, npy_file)
#     pth_path = '/data1/exp/ctdet/mobilenetv2/model_last.pth'
#     data_dict = decode_pth_model(pth_path)
#     # data_dict = decode_npy_model(npy_file)
#     keys = list(data_dict.keys())
#     # var_name_lst = [key for key in keys if 'pfld_inference' in key]
#     var_name_lst = keys
#
#     net = caffe.Net('./%s.prototxt' % model_name, caffe.TEST)
#
#     # idx_w_notBN = {'weight': 0, 'depthwise_weight': 0, 'bias': 1}
#     # idx_w_BN = {'running_mean': 0, 'running_var': 1}
#
#     for var_name in var_name_lst:
#         if 'bottleneck' in var_name:    # bottleneck layer
#             if 'conv' in var_name:
#                 layer_name = '/'.join(var_name.split('.')[2:-1])
#                 if 'weight' in var_name:
#                     weight = data_dict[var_name]
#                     save_conv2caffe(weights=weight,conv_param=net.params[layer_name])
#                 elif 'bias' in var_name:
#                     bias = data_dict[var_name]
#                     save_conv2caffe(biases=bias,conv_param=net.params[layer_name])
#             elif 'bn' in var_name:
#                 layer_name = map_torch_bn_layer_to_caffe_bn(var_name)
#                 if 'mean' in var_name:
#                     mean = data_dict[var_name]
#                     save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
#                 elif 'var' in var_name:
#                     var = data_dict[var_name]
#                     save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
#                 elif 'weight' in var_name:
#                     weight = data_dict[var_name]
#                     save_scale2caffe(weights=weight,scale_param=net.params[layer_name])
#                 elif 'bias' in var_name:
#                     bias = data_dict[var_name]
#                     save_scale2caffe(biases=bias,scale_param=net.params[layer_name])
#                 else:
#                     continue
#             else:
#                 continue
#         elif 'mapheatmap' in var_name:           # heatmap layer
#             layer_num = var_name.split('.')[-2]
#             if layer_num in ['0','4']:
#                 #conv with bias
#                 layer_name='/'.join(var_name.split('.')[1:-1])
#                 if 'weight' in var_name:
#                     weight = data_dict[var_name]
#                     save_conv2caffe(weights=weight,conv_param=net.params[layer_name])
#                 elif 'bias' in var_name:
#                     bias = data_dict[var_name]
#                     save_conv2caffe(biases=bias, conv_param=net.params[layer_name])
#
#             elif layer_num=='1':
#                 # bn
#                 if 'run' in var_name:
#                     layer_name = '/'.join(var_name.split('.')[1:3]+['0/BatchNorm'])
#                 else:
#                     layer_name = '/'.join(var_name.split('.')[1:3]+['0/scale'])
#                 if 'mean' in var_name:
#                     mean = data_dict[var_name]
#                     save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
#                 elif 'var' in var_name:
#                     var = data_dict[var_name]
#                     save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
#                 elif 'weight' in var_name:
#                     weight = data_dict[var_name]
#                     save_scale2caffe(weights=weight, scale_param=net.params[layer_name])
#                 elif 'bias' in var_name:
#                     bias = data_dict[var_name]
#                     save_scale2caffe(biases=bias, scale_param=net.params[layer_name])
#                 else:
#                     continue
#             elif layer_num=='2':
#                 # conv without bias
#                 layer_name='/'.join(var_name.split('.')[1:-1])
#                 if 'weight' in var_name:
#                     weight = data_dict[var_name]
#                     save_conv2caffe(weights=weight,conv_param=net.params[layer_name])
#             else:
#                 continue
#         elif 'conv1' in var_name:
#             if 'weight' in var_name:
#                 weight = data_dict[var_name]
#                 layer_name = var_name.split('.')[0]
#                 save_conv2caffe(weights=weight,conv_param= net.params[layer_name])
#         elif 'bn1' in var_name:
#             if 'run' in var_name:
#                 layer_name = 'conv1/BatchNorm'
#             else:
#                 layer_name = 'conv1/scale'
#             if 'mean' in var_name:
#                 mean = data_dict[var_name]
#                 save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
#             elif 'var' in var_name:
#                 var = data_dict[var_name]
#                 save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
#             elif 'weight' in var_name:
#                 weight = data_dict[var_name]
#                 save_scale2caffe(weights=weight, scale_param=net.params[layer_name])
#             elif 'bias' in var_name:
#                 bias = data_dict[var_name]
#                 save_scale2caffe(biases=bias, scale_param=net.params[layer_name])
#             else:
#                 continue
#
#
#     net.save('./%s.caffemodel' % model_name)
#
#
# def test_model(model_name):
#     import cv2
#     import torch
#     from mobilenetv2 import MobileNetV2
#     checkpoint_path = '/data1/exp/ctdet/mobilenetv2/model_last.pth'
#     image = cv2.imread('540e3f90874dfa66.jpg')
#     # image = np.random.randn(112, 112, 3)*255
#     # image = image.astype(np.uint8)
#     input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
#     # input = image.copy()[...,::-1]
#     input = cv2.resize(input, (512, 512))
#     # debug
#     # input = input[:8, :8, :]
#     input = input.astype(np.float32) / 256.0
#     input = np.expand_dims(input, 0)
#     torch_input = input.copy().transpose((0, 3, 1, 2))
#     tensor_input = torch.from_numpy(torch_input)
#     input_ = input.copy()
#     heads = {'hm': 6, 'wh': 2, 'reg': 2}
#     num_layers = 34
#     model3 = MobileNetV2(heads, head_conv=64)
#     model3.load_state_dict(torch.load(checkpoint_path)['state_dict'])
#
#     pytorch_result = model3(tensor_input)
#
#
#     net = caffe.Net('./%s.prototxt' % model_name, './%s.caffemodel' % model_name, caffe.TEST)
#     input_ = input.transpose((0, 3, 1, 2))
#
#     net.blobs['data'].data[...] = input_
#     output_ = net.forward()
#     # 把数据经过xxx层后的结果输出来
#     out = net.blobs['Convolution1'].data[0]
#     # print(output_)
#     keys = list(output_.keys())
#     print(output_[keys[0]].shape)
#     caffe_output = output_[keys[0]]
#
#     def cal_MPA(caffe_output, cmp_output):
#         try:
#             error = np.abs(caffe_output - cmp_output)
#         except:
#             cmp_output = cmp_output.transpose((0, 3, 1, 2))
#             error = np.abs(caffe_output - cmp_output)
#         zeros = np.zeros_like(error)
#         error = np.where(np.less(error, 1e-5), zeros, error)
#         print('error: ', np.sum(error))
#         MPA = np.max(error) / np.max(np.abs(cmp_output)) * 100.
#         print('MPA: %f' % MPA)
#
#     cmp_output = pytorch_result
#     cal_MPA(caffe_output, cmp_output)
#
#     bin_file = '/data2/SharedVMs/nfs_sync/model_speed_test/mobileResult.bin'
#     hisi_result = np.fromfile(bin_file, dtype=np.float32)
#     hisi_result = np.reshape(hisi_result, [1, 196])
#     cal_MPA(caffe_output, hisi_result)
#
#     caffe_output.astype(dtype=np.float32)
#     caffe_output.tofile('./data/caffe_varify_output.bin')
#
#
#
#     print('Done.')
#
#
# def main():
#     model_name = 'MobileNet_CenterNet'
#     # gen_pfld_prototxt(model_name=model_name)
#     npy_file = './mobilenet_centernet.npy'
#     # meta_file = './TF_model/model.meta'
#     # ckpt_file = './TF_model/model.ckpt-312'
#     # save_caffemodel(npy_file, model_name=model_name,
#     #                 meta_file=meta_file, ckpt_file=ckpt_file)
#     save_caffemodel(model_name)
#     test_model(model_name)
#
#
# if __name__ == '__main__':
#     gen_prototxt()
#     main()


# -*- coding: utf-8 -*-
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
from caffe import layers as L, params as P
import numpy as np
from utils import conv2d, depthwise_conv2d, bottleneck, heartmap, subnet, deconv

from utils import get_npy, decode_npy_model, decode_pth_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def mobilenetv2_centernet_inference(netspec, input_node):
    layer_cfg = [['conv2d', 32, 3, 2, 'relu', False, True, 'conv1'],
                 ['bottleneck_0_0', 32, 16, 1, 'LinearBottleneck0_0'],
                 ['bottleneck_1_0', 96, 24, 2, 'LinearBottleneck1_0'],
                 ['bottleneck_1_1', 144, 24, 1, 'LinearBottleneck1_1'],
                 ['bottleneck_2_0', 144, 32, 2, 'LinearBottleneck2_0'],
                 ['bottleneck_2_1', 192, 32, 1, 'LinearBottleneck2_1'],
                 ['bottleneck_3_0', 192, 64, 2, 'LinearBottleneck3_0'],
                 ['bottleneck_3_1', 384, 64, 1, 'LinearBottleneck3_1'],
                 ['bottleneck_3_2', 384, 64, 1, 'LinearBottleneck3_2'],
                 ['bottleneck_4_0', 384, 96, 1, 'LinearBottleneck4_0'],
                 ['bottleneck_4_1', 576, 96, 1, 'LinearBottleneck4_1'],
                 ['bottleneck_4_2', 576, 96, 1, 'LinearBottleneck4_2'],
                 ['bottleneck_5_0', 576, 160, 2, 'LinearBottleneck5_0'],
                 ['bottleneck_5_1', 960, 160, 1, 'LinearBottleneck5_1'],
                 ['bottleneck_6_0', 960, 320, 1, 'LinearBottleneck6_0']]
    deconv_cfg = [['deconv', 160, 'deconv1'],
                    ['deconv', 160, 'deconv2'],
                    ['deconv', 64, 'deconv3']]

    n = netspec
    blobs_lst = []
    layer = layer_cfg[0]
    n.conv = conv2d(n, input_node, num_output=32, kernel_size=3,
                    stride=2,
                    activation_fn='relu',
                    bias_term=False,
                    use_bn=True,
                    scope='conv1')
    resnet_block = [False,False,True,False,True,False,True,True,False,True,True,False,True,False]
    layer = layer_cfg[1]
    n.bottleneck_0_0 = bottleneck(n, n.conv, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[0])
    layer = layer_cfg[2]
    n.bottleneck_1_0 = bottleneck(n, n.bottleneck_0_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[1])
    layer = layer_cfg[3]
    n.bottleneck_1_1 = bottleneck(n, n.bottleneck_1_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[2])
    layer = layer_cfg[4]
    n.bottleneck_2_0 = bottleneck(n, n.bottleneck_1_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[3])
    layer = layer_cfg[5]
    n.bottleneck_2_1 = bottleneck(n, n.bottleneck_2_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[4])
    layer = layer_cfg[6]
    n.bottleneck_3_0 = bottleneck(n, n.bottleneck_2_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[5])
    layer = layer_cfg[7]
    n.bottleneck_3_1 = bottleneck(n, n.bottleneck_3_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[6])
    layer = layer_cfg[8]
    n.bottleneck_3_2 = bottleneck(n, n.bottleneck_3_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[7])
    layer = layer_cfg[9]
    n.bottleneck_4_0 = bottleneck(n, n.bottleneck_3_2, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[8])
    layer = layer_cfg[10]
    n.bottleneck_4_1 = bottleneck(n, n.bottleneck_4_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[9])
    layer = layer_cfg[11]
    n.bottleneck_4_2 = bottleneck(n, n.bottleneck_4_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[10])
    layer = layer_cfg[12]
    n.bottleneck_5_0 = bottleneck(n, n.bottleneck_4_2, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[11])
    layer = layer_cfg[13]
    n.bottleneck_5_1 = bottleneck(n, n.bottleneck_5_0, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[12])
    layer = layer_cfg[14]
    n.bottleneck_6_0 = bottleneck(n, n.bottleneck_5_1, c_e=layer[1], c_o=layer[2], stride=layer[3], scope=layer[4],
                                  resnet_block=resnet_block[13])

    deconv_layer = deconv_cfg[0]
    n.deconv1 = deconv(n,n.bottleneck_6_0,deconv_layer[1], scope=deconv_layer[2])
    deconv_layer = deconv_cfg[1]
    n.deconv2 = deconv(n, n.deconv1, deconv_layer[1], scope=deconv_layer[2])
    deconv_layer = deconv_cfg[2]
    n.deconv3 = deconv(n, n.deconv2, deconv_layer[1], scope=deconv_layer[2])

    n.hm, n.wh, n.reg = subnet(n, n.deconv3, c_i=64, scope='heatmap_layer')
    return n


def create_model(depth_coe=1.):
    n = caffe.NetSpec()

    n.data = L.Input(shape=[dict(dim=[1, 3, 512, 512])], ntop=1)
    n = mobilenetv2_centernet_inference(n, n.data)

    return n


def parse_caffemodel(caffemodel):
    MODEL_FILE = '/home/amax/workspace/pytorch_caffe/deploy.prototxt'
    # 预先训练好的caffe模型
    PRETRAIN_FILE = caffemodel

    # 保存参数的文件
    params_txt = 'params.txt'
    pf = open(params_txt, 'w')

    # 让caffe以测试模式读取网络参数
    net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

    # 遍历每一层
    for param_name in net.params.keys():
        # 权重参数
        weight = net.params[param_name][0].data
        # 偏置参数
        bias = net.params[param_name][1].data

        # 该层在prototxt文件中对应“top”的名称
        pf.write(param_name)
        pf.write('\n')

        # 写权重参数
        pf.write('\n' + param_name + '_weight:\n\n')
        # 权重参数是多维数组，为了方便输出，转为单列数组
        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%ff, ' % w)

        # 写偏置参数
        pf.write('\n\n' + param_name + '_bias:\n\n')
        # 偏置参数是多维数组，为了方便输出，转为单列数组
        bias.shape = (-1, 1)
        for b in bias:
            pf.write('%ff, ' % b)

        pf.write('\n\n')

    pf.close()

    print('--')


def gen_prototxt(model_name='new_MobileNet_CenterNet'):
    net = create_model()
    with open('%s.prototxt' % model_name, 'w') as f:
        f.write(str(net.to_proto()))


def save_conv2caffe(weights=None, biases=None, conv_param=None):
    if conv_param is not None:
        if biases is not None:
            conv_param[1].data[...] = biases
        if weights is not None:
            conv_param[0].data[...] = weights

def save_deconv2caffe(weights=None, biases=None, deconv_param=None):
    if deconv_param is not None:
        if biases is not None:
            deconv_param[1].data[...] = biases
        if weights is not None:
            deconv_param[0].data[...] = weights

def save_fc2caffe(weights, biases, fc_param):
    print(biases.size(), weights.size())
    print(fc_param[1].data.shape)
    print(fc_param[0].data.shape)
    fc_param[1].data[...] = biases
    fc_param[0].data[...] = weights


def save_bn2caffe(running_mean=None, running_var=None, bn_param=None):
    if bn_param is not None:
        if running_mean is not None:
            bn_param[0].data[...] = running_mean
        if running_var is not None:
            bn_param[1].data[...] = running_var
        bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights=None, biases=None, scale_param=None):
    if scale_param is not None:
        if biases is not None:
            scale_param[1].data[...] = biases
        if weights is not None:
            scale_param[0].data[...] = weights


def map_torch_bn_layer_to_caffe_bn(bn_layer_name):
    layer_name = bn_layer_name.replace('bn', 'conv')
    lst = layer_name.split('.')
    if 'run' in layer_name:
        new_lst = lst[2:-1] + ['BatchNorm']
    else:
        new_lst = lst[2:-1] + ['scale']
    caffe_bn_layer_name = '/'.join(new_lst)
    return caffe_bn_layer_name


def save_caffemodel(model_name, pth_path=None):
    # if meta_file is not None and ckpt_file is not None:
    #     convert_meta_to_npy(meta_file, ckpt_file, npy_file)
    pth_path = '/data1/exp/ctdet/default/model_best.pth'
    data_dict = decode_pth_model(pth_path)
    # data_dict = decode_npy_model(npy_file)
    keys = list(data_dict.keys())
    # var_name_lst = [key for key in keys if 'pfld_inference' in key]
    var_name_lst = keys

    net = caffe.Net('./%s.prototxt' % model_name, caffe.TEST)

    # idx_w_notBN = {'weight': 0, 'depthwise_weight': 0, 'bias': 1}
    # idx_w_BN = {'running_mean': 0, 'running_var': 1}

    for var_name in var_name_lst:
        if 'bottleneck' in var_name:  # bottleneck layer
            if 'conv' in var_name:
                layer_name = '/'.join(var_name.split('.')[2:-1])
                if 'weight' in var_name:
                    weight = data_dict[var_name]
                    save_conv2caffe(weights=weight, conv_param=net.params[layer_name])
                elif 'bias' in var_name:
                    bias = data_dict[var_name]
                    save_conv2caffe(biases=bias, conv_param=net.params[layer_name])
            elif 'bn' in var_name:
                layer_name = map_torch_bn_layer_to_caffe_bn(var_name)
                if 'mean' in var_name:
                    mean = data_dict[var_name]
                    save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
                elif 'var' in var_name:
                    var = data_dict[var_name]
                    save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
                elif 'weight' in var_name:
                    weight = data_dict[var_name]
                    save_scale2caffe(weights=weight, scale_param=net.params[layer_name])
                elif 'bias' in var_name:
                    bias = data_dict[var_name]
                    save_scale2caffe(biases=bias, scale_param=net.params[layer_name])
                else:
                    continue
            else:
                continue
        elif 'mapheatmap' in var_name:  # heatmap layer
            layer_num = var_name.split('.')[-2]
            if layer_num in ['0', '4']:
                # conv with bias
                layer_name = '/'.join(var_name.split('.')[1:-1])
                if 'weight' in var_name:
                    weight = data_dict[var_name]
                    save_conv2caffe(weights=weight, conv_param=net.params[layer_name])
                elif 'bias' in var_name:
                    bias = data_dict[var_name]
                    save_conv2caffe(biases=bias, conv_param=net.params[layer_name])

            elif layer_num == '1':
                # bn
                if 'run' in var_name:
                    layer_name = '/'.join(var_name.split('.')[1:3] + ['0/BatchNorm'])
                else:
                    layer_name = '/'.join(var_name.split('.')[1:3] + ['0/scale'])
                if 'mean' in var_name:
                    mean = data_dict[var_name]
                    save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
                elif 'var' in var_name:
                    var = data_dict[var_name]
                    save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
                elif 'weight' in var_name:
                    weight = data_dict[var_name]
                    save_scale2caffe(weights=weight, scale_param=net.params[layer_name])
                elif 'bias' in var_name:
                    bias = data_dict[var_name]
                    save_scale2caffe(biases=bias, scale_param=net.params[layer_name])
                else:
                    continue
            elif layer_num == '2':
                # conv without bias
                layer_name = '/'.join(var_name.split('.')[1:-1])
                if 'weight' in var_name:
                    weight = data_dict[var_name]
                    save_conv2caffe(weights=weight, conv_param=net.params[layer_name])
            else:
                continue
        elif 'conv1' in var_name:
            if 'weight' in var_name:
                weight = data_dict[var_name]
                layer_name = var_name.split('.')[0]
                save_conv2caffe(weights=weight, conv_param=net.params[layer_name])
        elif 'bn1' in var_name:
            if 'run' in var_name:
                layer_name = 'conv1/BatchNorm'
            else:
                layer_name = 'conv1/scale'
            if 'mean' in var_name:
                mean = data_dict[var_name]
                save_bn2caffe(running_mean=mean, bn_param=net.params[layer_name])
            elif 'var' in var_name:
                var = data_dict[var_name]
                save_bn2caffe(running_var=var, bn_param=net.params[layer_name])
            elif 'weight' in var_name:
                weight = data_dict[var_name]
                save_scale2caffe(weights=weight, scale_param=net.params[layer_name])
            elif 'bias' in var_name:
                bias = data_dict[var_name]
                save_scale2caffe(biases=bias, scale_param=net.params[layer_name])
            else:
                continue
        # elif 'deconv' in var_name:
        #     layer_name = var_name.split('.')[0]
        #     if 'weight' in var_name:
        #         weight = data_dict[var_name]
        #         save_deconv2caffe(weights=weight,deconv_param=net.params[layer_name])
        #     else:
        #         bias = data_dict[var_name]
        #         save_deconv2caffe(biases=bias,deconv_param=net.params[layer_name])

        elif var_name.split('.')[0] in ['wh','hm','reg']:
            layer_name = '/'.join(['heatmap_layer']+var_name.split('.')[:2])
            if 'weight' in var_name:
                weight=data_dict[var_name]
                save_conv2caffe(weights=weight,conv_param=net.params[layer_name])
            else:
                bias = data_dict[var_name]
                save_conv2caffe(biases=bias,conv_param=net.params[layer_name])
        weight1 = data_dict['deconv1.weight']
        bias1 = data_dict['deconv1.bias']
        net.params['deconv1'][0].data[...]=weight1
        net.params['deconv1'][1].data[...]=bias1
        weight2 = data_dict['deconv2.weight']
        bias2 = data_dict['deconv2.bias']
        net.params['deconv2'][0].data[...] = weight2
        net.params['deconv2'][1].data[...] = bias2
        weight3 = data_dict['deconv3.weight']
        bias3 = data_dict['deconv3.bias']
        net.params['deconv3'][0].data[...] = weight3
        net.params['deconv3'][1].data[...] = bias3

    net.save('./%s.caffemodel' % model_name)


def test_model(model_name):
    import cv2
    import torch
    from mobilenetv2 import MobileNetV2
    checkpoint_path = '/data1/exp/ctdet/default/model_best.pth'
    image = cv2.imread('540e3f90874dfa66.jpg')
    # image = np.random.randn(112, 112, 3)*255
    # image = image.astype(np.uint8)
    input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # input = image.copy()[...,::-1]
    input = cv2.resize(input, (512, 512))
    # debug
    # input = input[:8, :8, :]
    input = input.astype(np.float32) / 256.0
    input = np.expand_dims(input, 0)
    torch_input = input.copy().transpose((0, 3, 1, 2))
    tensor_input = torch.from_numpy(torch_input)
    input_ = input.copy()
    heads = {'hm': 6, 'wh': 2, 'reg': 2}
    num_layers = 34
    model3 = MobileNetV2(heads, head_conv=64)
    data_dict = torch.load(checkpoint_path)['state_dict']
    # data_dict.pop('bn1.weight')
    # data_dict.pop('bn1.bias')
    model3.load_state_dict(data_dict)
    model3.train(False)

    pytorch_result = model3(tensor_input)

    net = caffe.Net('./%s.prototxt' % model_name, './%s.caffemodel' % model_name, caffe.TEST)
    input_ = input.transpose((0, 3, 1, 2))

    net.blobs['data'].data[...] = input_
    output_ = net.forward()
    # output_ = net.forward(end='deconv1')  # 获取指定层的输出
    # print(output_)
    keys = list(output_.keys())
    print(output_[keys[0]].shape)
    caffe_output = output_[keys[0]]

    def cal_MPA(caffe_output, cmp_output):
        try:
            error = np.abs(caffe_output - cmp_output)
        except:
            cmp_output = cmp_output.transpose((0, 3, 1, 2))
            error = np.abs(caffe_output - cmp_output)
        zeros = np.zeros_like(error)
        error = np.where(np.less(error, 1e-5), zeros, error)
        print('error: ', np.sum(error))
        MPA = np.max(error) / np.max(np.abs(cmp_output)) * 100.
        print('MPA: %f' % MPA)

    # cmp_output = pytorch_result.cpu().detach().numpy()
    # cal_MPA(caffe_output, cmp_output)
    for k,val in output_.items():
        cmp_output = pytorch_result[0][k].cpu().detach().numpy()
        cal_MPA(val, cmp_output)

    # bin_file = '/data2/SharedVMs/nfs_sync/model_speed_test/mobileResult.bin'
    # hisi_result = np.fromfile(bin_file, dtype=np.float32)
    # hisi_result = np.reshape(hisi_result, [1, 196])
    # cal_MPA(caffe_output, hisi_result)
    #
    # caffe_output.astype(dtype=np.float32)
    # caffe_output.tofile('./data/caffe_varify_output.bin')

    print('Done.')


def main():
    model_name = 'new_MobileNet_CenterNet'
    # gen_pfld_prototxt(model_name=model_name)
    # npy_file = './mobilenet_centernet.npy'
    # meta_file = './TF_model/model.meta'
    # ckpt_file = './TF_model/model.ckpt-312'
    # save_caffemodel(npy_file, model_name=model_name,
    #                 meta_file=meta_file, ckpt_file=ckpt_file)
    save_caffemodel(model_name)
    test_model(model_name)


if __name__ == '__main__':
    gen_prototxt()
    main()
