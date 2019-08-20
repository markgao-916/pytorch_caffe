#
# from collections import OrderedDict
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import numpy as np
#
#
# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# class LinearBottleneck(nn.Module):
#     def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU):
#         super(LinearBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(inplanes * t)
#         self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
#                                groups=inplanes * t)
#         self.bn2 = nn.BatchNorm2d(inplanes * t)
#         self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(outplanes)
#         self.activation = activation(inplace=True)
#         self.stride = stride
#         self.t = t
#         self.inplanes = inplanes
#         self.outplanes = outplanes
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.activation(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.stride == 1 and self.inplanes == self.outplanes:
#             out += residual
#         # print('out.shape', out.shape)
#         return out
#
#
# class MapHeatmap(nn.Module):
#     def __init__(self, input_channels, heads, head_conv):
#         super(MapHeatmap,self).__init__()
#         self.heads = heads
#         self.head_conv = head_conv
#         final_kernel = 1
#         for head in self.heads:
#             classes = self.heads[head]
#             fc = nn.Sequential(
#                 nn.Conv2d(input_channels, input_channels,
#                           kernel_size=3, padding=1, bias=True, groups=input_channels),
#                 nn.BatchNorm2d(input_channels),
#                 nn.Conv2d(input_channels, self.head_conv, kernel_size=1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.head_conv, classes,
#                           kernel_size=final_kernel, stride=1,
#                           padding=final_kernel // 2, bias=True))
#             if 'hm' in head:
#                 fc[-1].bias.data.fill_(-2.19)
#             else:
#                 self.fill_fc_weights(fc)
#
#             self.__setattr__(head, fc)
#         pass
#
#     def forward(self, x):
#         hm_x = self.__getattr__('hm')(x)
#         wh_x = self.__getattr__('wh')(x)
#         reg_x = self.__getattr__('reg')(x)
#         return {'hm':hm_x,'wh':wh_x,'reg':reg_x}
#
#         pass
#
#     def fill_fc_weights(self,layers):
#         for m in layers.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
# class MobileNetV2(nn.Module):
#     """MobileNet2 implementation.
#     """
#
#     def __init__(self, heads, head_conv=64, scale=1.0, in_channels=3, activation=nn.ReLU, pretrained=True):
#         """
#         MobileNet2 constructor.
#         :param in_channels: (int, optional): number of channels in the input tensor.
#                 Default is 3 for RGB image inputs.
#         :param input_size:
#         :param num_classes: number of classes to predict. Default
#                 is 1000 for ImageNet.
#         :param scale:
#         :param t:
#         :param activation:
#         """
#
#         super(MobileNetV2, self).__init__()
#
#         self.scale = scale
#         self.t = 6
#         self.activation_type = activation
#         self.activation = activation(inplace=True)
#
#         self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
#         # assert (input_size % 32 == 0)
#
#         self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
#         self.n = [1, 1, 2, 2, 3, 3, 2, 1]
#         self.s = [2, 1, 2, 2, 2, 1, 2, 1]
#         self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
#         self.bn1 = nn.BatchNorm2d(self.c[0])
#         self.bottlenecks = self._make_bottlenecks()
#         self.adop_layer_index = [2,4,6]
#         self.mapheatmap = self._make_mapheatmap(heads,head_conv)
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
#         modules = OrderedDict()
#         stage_name = "LinearBottleneck{}".format(stage)
#
#         # First module is the only one utilizing stride
#         first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
#                                         activation=self.activation_type)
#         modules[stage_name + "_0"] = first_module
#
#         # add more LinearBottleneck depending on number of repeats
#         for i in range(n - 1):
#             name = stage_name + "_{}".format(i + 1)
#             module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
#                                       activation=self.activation_type)
#             modules[name] = module
#
#         return nn.Sequential(modules)
#
#     def _make_bottlenecks(self):
#         modules = OrderedDict()
#         stage_name = "Bottlenecks"
#
#         # First module is the only one with t=1
#         bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
#                                        stage=0)
#         modules[stage_name + "_0"] = bottleneck1
#
#         # add more LinearBottleneck depending on number of repeats
#         for i in range(1, len(self.c) - 1):
#             name = stage_name + "_{}".format(i)
#             module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
#                                       stride=self.s[i + 1],
#                                       t=self.t, stage=i)
#             modules[name] = module
#
#         return nn.Sequential(modules)
#
#     def _make_mapheatmap(self, heads,head_conv):
#         modules = OrderedDict()
#         stage_name = "MapHeatmap"
#         # add subnet
#         for i in self.adop_layer_index:
#             name = stage_name + "_{}".format(i)
#             input_channels = self.num_of_channels[i+1]
#             module = MapHeatmap(input_channels, heads, head_conv)
#             modules[name] = module
#
#         return nn.Sequential(modules)
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#         x_in = x
#         x = self.bn1(x)
#         x = self.activation(x)
#         # print('x.shape:', x.shape)
#
#         # x = self.bottlenecks(x)
#         x = self.bottlenecks.Bottlenecks_0(x)
#         x1 = self.bottlenecks.Bottlenecks_1(x)
#         x2 = self.bottlenecks.Bottlenecks_2(x1)
#         x3 = self.bottlenecks.Bottlenecks_3(x2)
#         x4 = self.bottlenecks.Bottlenecks_4(x3)
#         x5 = self.bottlenecks.Bottlenecks_5(x4)
#         x6 = self.bottlenecks.Bottlenecks_6(x5)
#         result = []
#         result.append(self.mapheatmap.MapHeatmap_2(x2))
#         result.append(self.mapheatmap.MapHeatmap_4(x4))
#         result.append(self.mapheatmap.MapHeatmap_6(x6))
#         # return result
#         return x_in
#
# def test():
#     input_channels=3
#     heads = {'hm': 6, 'wh': 2, 'reg': 2}
#     head_conv=128
#     net = MapHeatmap(input_channels, heads, head_conv)
#     x = torch.randn(2,3,224,224)
#     y = net(x)
#     print(y.size())
#
# def get_mobilenetv2(num_layers, heads, head_conv=256, down_ratio=4,pretrained=True):
#     model = MobileNetV2(heads, head_conv)
#     return model
#
# if __name__ == "__main__":
#     import cv2
#
#     heads = {'hm': 6, 'wh': 2, 'reg': 2}
#     num_layers = 34
#     model3 = MobileNetV2(heads, head_conv=64)
#     model3.load_state_dict(torch.load('/data1/exp/ctdet/coco_dla_0.5channel/model_last.pth')['state_dict'])
#     image = cv2.imread('540e3f90874dfa66.jpg')
#     # image = np.random.randn(112, 112, 3)*255
#     # image = image.astype(np.uint8)
#     input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
#     # input = image.copy()[...,::-1]
#     input = cv2.resize(input, (512, 512))
#     input = np.transpose(input,[2,0,1])
#     # debug
#     # input = input[:8, :8, :]
#     input = input.astype(np.float32) / 256.0
#     input = np.expand_dims(input, 0)
#     tensor_input = torch.from_numpy(input)
#     x = torch.randn(1, 3, 512, 512)
#
#     result = (model3(tensor_input))
#     print('---')
#
#
#
#
# # test()





from collections import OrderedDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t,track_running_stats=True)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t,track_running_stats=True)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes,track_running_stats=True)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        # print('out.shape', out.shape)
        return out


class MapHeatmap(nn.Module):
    def __init__(self, input_channels, heads, head_conv):
        super(MapHeatmap,self).__init__()
        self.heads = heads
        self.head_conv = head_conv
        final_kernel = 1
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=3, padding=1, bias=True, groups=input_channels),
                nn.BatchNorm2d(input_channels),
                nn.Conv2d(input_channels, self.head_conv, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes,
                          kernel_size=final_kernel, stride=1,
                          padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        pass

    def forward(self, x):
        hm_x = self.__getattr__('hm')(x)
        wh_x = self.__getattr__('wh')(x)
        reg_x = self.__getattr__('reg')(x)
        return {'hm':hm_x,'wh':wh_x,'reg':reg_x}

        pass

    def fill_fc_weights(self,layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class MobileNetV2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, heads, head_conv=256, scale=1.0, in_channels=3, activation=nn.ReLU, pretrained=True):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNetV2, self).__init__()

        self.scale = scale
        self.t = 6
        self.activation_type = activation
        self.activation = activation(inplace=True)
        channels = [16, 32, 64, 128, 256, 512]
        final_kernel = 1
        self.first_level = 2

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        self.deconv_channels_num = [320,160,64]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 2, 3, 3, 2, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0],track_running_stats=True)
        self.bottlenecks = self._make_bottlenecks()

        self.deconv1 = nn.ConvTranspose2d(320,160,kernel_size=4,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(160, 160, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(160, 64, kernel_size=4, stride=2, padding=1)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    self.fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    self.fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self.init_params()

    def fill_fc_weights(self,layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.bottlenecks(x)
        decon = self.deconv1(x)
        decon = self.deconv2(decon)
        decon = self.deconv3(decon)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(decon)
        return [z]
        # return decon

def test():
    input_channels=3
    heads = {'hm': 6, 'wh': 2, 'reg': 2}
    head_conv=128
    net = MapHeatmap(input_channels, heads, head_conv)
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

def get_mobilenetv2(num_layers, heads, head_conv=64, down_ratio=4,pretrained=True):
    model = MobileNetV2(heads, head_conv)
    return model

if __name__ == "__main__":
    """Testing
    """
    # test()
    # model1 = MobileNetV2()
    # print(model1)
    # model2 = MobileNetV2(scale=0.35)
    # print(model2)
    heads = {'hm': 6, 'wh': 2, 'reg': 2}
    num_layers = 34
    model3 = MobileNetV2(heads,head_conv=64)
    print(model3)
    x = torch.randn(1, 3, 512, 512)
    print(model3(x))
    # model4_size = 32 * 10
    # model4 = MobileNetV2(input_size=model4_size, num_classes=10)
    # print(model4)
    # x2 = torch.randn(1, 3, model4_size, model4_size)
    # print(model4(x2))
    # model5 = MobileNetV2(input_size=196, num_classes=10)
    # x3 = torch.randn(1, 3, 196, 196)
    # print(model5(x3))  # fail



# test()