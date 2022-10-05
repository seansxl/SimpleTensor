import torch
import torch.nn as nn
from collections import OrderedDict


"""
参考：https://www.cnblogs.com/silence-cho/p/13976981.html
"""

class ConvolutionalLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = True):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(nn.Module):
    def __init__(self,
                 in_channels: int):
        super(ResidualLayer, self).__init__()
        c_out = in_channels // 2
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, c_out, 1, 1, 0),
            ConvolutionalLayer(c_out, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.sub_module(x)


class ConvolutionalSetLayer(nn.Module):
    def __init__(self,
                 filters_list: list,
                 in_filters: int,
                 out_filter: int):
        super(ConvolutionalSetLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_filters, filters_list[0], 1, 1, 0),

            ConvolutionalLayer(filters_list[0], filters_list[1], 3, 1, 1),
            ConvolutionalLayer(filters_list[1], filters_list[0], 1, 1, 0),
            ConvolutionalLayer(filters_list[0], filters_list[1], 3, 1, 1),
            ConvolutionalLayer(filters_list[1], filters_list[0], 1, 1, 0),

            ConvolutionalLayer(filters_list[0], filters_list[1], 3, 1, 1),
            nn.Conv2d(filters_list[1], out_filter, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)

    def __getitem__(self, key):
        return self.sub_module[key]

    def __iter__(self):
        return self


class DownSamplingLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(DownSamplingLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class UpSamplingLayer(nn.Module):
    def __init__(self):
        super(UpSamplingLayer, self).__init__()

    def forward(self, x):
        return nn.Upsample(scale_factor=2, mode='nearest')(x)


class Darknet(nn.Module):
    def __init__(self, layers):
        super(Darknet, self).__init__()
        self.inplanes = 32
        self.conv1 = ConvolutionalLayer(3, self.inplanes, 3, 1, 1)

        self.layer1 = self.__make_layers([32, 64], layers[0])
        self.layer2 = self.__make_layers([64, 128], layers[1])
        self.layer3 = self.__make_layers([128, 256], layers[2])
        self.layer4 = self.__make_layers([256, 512], layers[3])
        self.layer5 = self.__make_layers([512, 1024], layers[4])

    def __make_layers(self, planes: list, blocks: list):
        layers = [('down_sample', DownSamplingLayer(self.inplanes, planes[1]))]
        self.inplanes = planes[1]
        for i in range(blocks):
            layers.append(('residual_{}'.format(i), ResidualLayer(planes[1])))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.layer1(conv1)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5


class YoloNet(nn.Module):
    def __init__(self, anchors_mask=None, num_classes=None):
        super(YoloNet, self).__init__()
        self.backbone = Darknet(layers=[1, 2, 8, 8, 4])
        self.last_layer0 = self.__make_last_layer([512, 1024], 1024, 255)

        self.last_layer1_upsample = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpSamplingLayer()
        )

        self.last_layer1 = self.__make_last_layer([256, 512], 768, 255)

        self.last_layer2_upsample = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpSamplingLayer()
        )
        self.last_layer2 = nn.Sequential(
            ConvolutionalSetLayer([128, 256], 384, 255)
        )

    def __make_last_layer(self, filters_list: list, in_filters: int, out_filter: int):
        return ConvolutionalSetLayer(filters_list, in_filters, out_filter)

    def forward(self, x):
        h_52, h_26, h_13 = self.backbone(x)
        out0_branch = self.last_layer0[:5](h_13)
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_upsample(out0_branch)
        x1_in = torch.concat([h_26, x1_in], dim=1)
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_layer2_upsample(out1_branch)
        x2_in = torch.concat([h_52, x2_in], dim=1)
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    net = YoloNet()
    out0, out1, out2 = net(x)
    print(out0.shape)
    print(out1.shape)
    print(out2.shape)
