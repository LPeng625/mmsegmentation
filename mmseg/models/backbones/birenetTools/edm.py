import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import sys

from .ops import Conv2d
from .config import config_model

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True,
                 *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 64, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


from mmcv.ops import DeformConv2dPack


class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1, is_ori=True):
        super(PDCBlock, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        # TODO cv 改成可变形卷积
        if (pdc == 'cv'):
            print('F.conv2d')
            self.conv1 = Conv2d(F.conv2d, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
            # print('DeformConv2dPack')
            # self.conv1 = DeformConv2dPack(in_channels=inplane, out_channels=inplane, kernel_size=3, padding=1,
            #                               groups=inplane, bias=False, stride=1, dilation=1)
        else:
            self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)
        # 原来的结构：add，新结构：concatenate
        self.is_ori = is_ori
        if not self.is_ori:
            self.conv3 = nn.Conv2d(2 * inplane, ouplane, kernel_size=1, padding=0)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        if self.is_ori:
            y = self.conv2(y)
            if self.stride > 1:
                x = self.shortcut(x)
            y = y + x
        else:
            y = torch.cat((x, y), 1)
            y = self.conv3(y)
        return y


class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


#
# # densenet的形式
# class Transition(nn.Module):
#     def __init__(self, nChannels, nOutChannels):
#         super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(nChannels)
#         self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
#                                bias=False)
#
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = F.avg_pool2d(out, 2)
#         return out
#
# # PDC模块中加入dense连接，并改成深度可分离卷积
# class PDC_Dense_Block(nn.Module):
#     def __init__(self, pdc, nChannels, growthRate):
#         super(PDC_Dense_Block, self).__init__()
#
#         interChannels = 2 * growthRate
#         self.bn1 = nn.BatchNorm2d(nChannels)
#         self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
#                                bias=False)
#
#         self.bn2 = nn.BatchNorm2d(interChannels)
#         self.conv2 = ops.Conv2d(pdc, interChannels, interChannels, kernel_size=3, padding=1, groups=interChannels,
#                                 bias=False)
#
#         self.bn3 = nn.BatchNorm2d(interChannels)
#         self.conv3 = nn.Conv2d(interChannels, growthRate, kernel_size=1, padding=0, bias=False)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out = torch.cat((x, out), 1)
#
#         return out
#
# # densenet结构
# class PiDiNet(nn.Module):
#     def __init__(self, inplane, pdcs, dil=None, sa=False):
#         super(PiDiNet, self).__init__()
#         self.sa = sa
#         if dil is not None:
#             assert isinstance(dil, int), 'dil should be an int'
#         self.dil = dil
#
#         # self.fuseplanes = []
#
#         self.inplane = inplane
#         # if convert:
#         #     if pdcs[0] == 'rd':
#         #         init_kernel_size = 5
#         #         init_padding = 2
#         #     else:
#         #         init_kernel_size = 3
#         #         init_padding = 1
#         #     self.init_block = nn.Conv2d(3, self.inplane,
#         #                                 kernel_size=init_kernel_size, padding=init_padding, bias=False)
#         #     block_class = PDCBlock_converted
#         # else:
#         growthRate = 12
#         nChannels = 2 * growthRate
#         nDenseBlocks = 4
#         reduction = 0.5
#
#         # self.trans_init = Transition(3, nChannels)
#         self.trans_init = nn.Conv2d(3, nChannels, 1, bias=False)
#
#         self.dense1 = self._make_dense([pdcs[0], pdcs[1], pdcs[2], pdcs[3]], nChannels, growthRate, nDenseBlocks)
#         nChannels += nDenseBlocks * growthRate
#         nOutChannels = int(math.floor(nChannels * reduction))
#         self.trans1 = Transition(nChannels, nOutChannels)
#
#         nChannels = nOutChannels
#         self.dense2 = self._make_dense([pdcs[4], pdcs[5], pdcs[6], pdcs[7]], nChannels, growthRate, nDenseBlocks)
#         nChannels += nDenseBlocks * growthRate
#         nOutChannels = int(math.floor(nChannels * reduction))
#         self.trans2 = Transition(nChannels, nOutChannels)
#
#         nChannels = nOutChannels
#         self.dense3 = self._make_dense([pdcs[8], pdcs[9], pdcs[10], pdcs[11]], nChannels, growthRate, nDenseBlocks)
#         nChannels += nDenseBlocks * growthRate
#         nOutChannels = int(math.floor(nChannels * reduction))
#         self.trans3 = Transition(nChannels, nOutChannels)
#
#         nChannels = nOutChannels
#         self.dense4 = self._make_dense([pdcs[12], pdcs[13], pdcs[14], pdcs[15]], nChannels, growthRate, nDenseBlocks)
#         nChannels += nDenseBlocks * growthRate
#         nOutChannels = int(math.floor(nChannels * reduction))
#         self.trans4 = Transition(nChannels, nOutChannels)
#
#     def _make_dense(self, pdcs, nChannels, growthRate, nDenseBlocks):
#         layers = []
#         for i in range(int(nDenseBlocks)):
#             layers.append(PDC_Dense_Block(pdcs[i], nChannels, growthRate))
#             nChannels += growthRate
#         return nn.Sequential(*layers)
#
#     def get_weights(self):
#         conv_weights = []
#         bn_weights = []
#         relu_weights = []
#         for pname, p in self.named_parameters():
#             if 'bn' in pname:
#                 bn_weights.append(p)
#             elif 'relu' in pname:
#                 relu_weights.append(p)
#             else:
#                 conv_weights.append(p)
#
#         return conv_weights, bn_weights, relu_weights
#
#     def forward(self, x):
#         x = self.trans_init(x)
#
#         x1_out = self.dense1(x)
#         x1 = self.trans1(x1_out)
#
#         x2_out = self.dense2(x1)
#         x2 = self.trans2(x2_out)
#
#         x3_out = self.dense3(x2)
#         x3 = self.trans3(x3_out)
#
#         x4_out = self.dense4(x3)
#         x4 = self.trans4(x4_out)
#
#         x_fuses = [x1, x2, x3, x4]
#
#         return x_fuses


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.reduction = nn.Linear(4 * dim, ratio * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B,C,H,W
        """

        B, C, H, W = x.shape

        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)  # B H/2*W/2 4*C
        x = self.reduction(x)  # B H/2*W/2 2*C

        x = x.view(B, self.ratio * C, H // 2, W // 2)

        return x


def pidinet_init(str='carv4', init_stride=1, is_ori=True, inplane=64):
    pdcs = config_model(str)
    return PiDiNet(inplane=inplane, pdcs=pdcs, dil=30, sa=True, init_stride=init_stride, is_ori=is_ori)


# 通道注意力
class SELayer(nn.Module):
    def __init__(self):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(3, 3 // 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3 // 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y


def CNN(x):
    # x = image.resize
    # print(x.shape)
    # x = F.interpolate(x, scale_factor=0.5)
    # print(x.shape)
    # cnn = extract_param.CNN()
    cnn = SELayer()
    cnn.cuda()
    param = cnn(x)
    return param


def init_weights(m):
    """
    初始化参数
    Args:
        m: 网络层
    Returns:
        初始化权重的网络层
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# TODO 空洞卷积金字塔空间注意力模块 cat拼接空洞卷积+SAM
class DSAM_Concat1(nn.Module):

    def __init__(self, channel):
        super(DSAM_Concat1, self).__init__()

        self.dilate1 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bn=True, relu=True)
        self.dilate2 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=2, dilation=2, bn=True, relu=True)
        self.dilate3 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=4, dilation=4, bn=True, relu=True)
        self.dilate4 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=8, dilation=8, bn=True, relu=True)

        self.convInit = ConvBNReLU(in_channels=2 * channel, out_channels=channel, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.convFinal = ConvBNReLU(in_channels=5 * channel, out_channels=channel, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.convInit(x)
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        y = torch.cat([x, dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        out = self.convFinal(y)
        out = self.SpatialGate(out) + out

        return out


# TODO 空洞卷积金字塔空间注意力模块 SAM + cat拼接空洞卷积
class DSAM_Concat2(nn.Module):

    def __init__(self, channel):
        super(DSAM_Concat2, self).__init__()

        self.dilate1 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bn=True, relu=True)
        self.dilate2 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=2, dilation=2, bn=True, relu=True)
        self.dilate3 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=4, dilation=4, bn=True, relu=True)
        self.dilate4 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=8, dilation=8, bn=True, relu=True)

        self.convInit = ConvBNReLU(in_channels=2 * channel, out_channels=channel, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.convFinal = ConvBNReLU(in_channels=5 * channel, out_channels=channel, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.convInit(x)
        x = self.SpatialGate(x) + x
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        y = torch.cat([x, dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        out = self.convFinal(y)

        return out


# TODO 空洞卷积金字塔空间注意力模块  cat拼接空洞卷积+SAM
class DSAM_Add1(nn.Module):

    def __init__(self, channel):
        super(DSAM_Add1, self).__init__()

        self.dilate1 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bn=True, relu=True)
        self.dilate2 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=2, dilation=2, bn=True, relu=True)
        self.dilate3 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=4, dilation=4, bn=True, relu=True)
        self.dilate4 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=8, dilation=8, bn=True, relu=True)

        self.convInit = ConvBNReLU(in_channels=2 * channel, out_channels=channel, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.convFinal = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.convInit(x)
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        x = dilate1_out + dilate2_out + dilate3_out + dilate4_out + x

        out = self.convFinal(x)
        out = self.SpatialGate(out)

        out += x

        return out


# TODO 空洞卷积金字塔空间注意力模块 SAM + add拼接空洞卷积
class DSAM_Add2(nn.Module):

    def __init__(self, channel):
        super(DSAM_Add2, self).__init__()

        self.dilate1 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bn=True, relu=True)
        self.dilate2 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=2, dilation=2, bn=True, relu=True)
        self.dilate3 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=4, dilation=4, bn=True, relu=True)
        self.dilate4 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=8, dilation=8, bn=True, relu=True)

        self.convInit = ConvBNReLU(in_channels=2 * channel, out_channels=channel, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.convFinal = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.convInit(x)
        x = self.SpatialGate(x)
        x += x
        out = self.convFinal(x)
        dilate1_out = self.dilate1(out)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + x

        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# TODO SAM CBAM空间注意力部分
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvBNReLU(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bn=True)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


# TODO 边缘提取模块
class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False, init_stride=1, is_ori=True):
        super(PiDiNet, self).__init__()

        self.fuseplanes = []
        self.inplane = inplane

        block_class = PDCBlock
        self.EDM_has_DSAM_Concat1 = True  # TODO 空洞卷积金字塔空间注意力模块 cat拼接空洞卷积+SAM
        self.EDM_has_DSAM_Concat2 = False  # TODO 空洞卷积金字塔空间注意力模块 cat拼接空洞卷积+SAM
        self.EDM_has_DSAM_Add1 = False  # TODO 空洞卷积金字塔空间注意力模块 add拼接空洞卷积+SAM
        self.EDM_has_DSAM_Add2 = False  # TODO 空洞卷积金字塔空间注意力模块 SAM+add拼接空洞卷积

        self.init_block = block_class(pdcs[0], 3, self.inplane, stride=2, is_ori=is_ori)
        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane, is_ori=is_ori)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane, is_ori=is_ori)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane, is_ori=is_ori)
        self.fuseplanes.append(self.inplane)  # C

        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2, is_ori=is_ori)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane, is_ori=is_ori)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane, is_ori=is_ori)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane, is_ori=is_ori)
        self.fuseplanes.append(self.inplane)  # 2C

        self.conv_reduces = nn.ModuleList()

        for i in range(2):
            self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        if self.EDM_has_DSAM_Concat1:
            self.DSAM = DSAM_Concat1(inplane)
        if self.EDM_has_DSAM_Concat2:
            self.DSAM = DSAM_Concat2(inplane)
        if self.EDM_has_DSAM_Add1:
            self.DSAM = DSAM_Add1(inplane)
        if self.EDM_has_DSAM_Add2:
            self.DSAM = DSAM_Add2(inplane)

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        H, W = H // 2, W // 2

        x1 = self.init_block(x)  # x torch.Size([2, 3, 1024, 1024]) x1 torch.Size([2, 64, 512, 512])
        x1 = self.block1_1(x1)  # x1 torch.Size([2, 64, 512, 512])
        x1 = self.block1_2(x1)  # x1 torch.Size([2, 64, 512, 512])
        x1 = self.block1_3(x1)  # x1 torch.Size([2, 64, 512, 512])

        x2 = self.block2_1(x1)  # x2 torch.Size([2, 128, 256, 256])
        x2 = self.block2_2(x2)  # x2 torch.Size([2, 128, 256, 256])
        x2 = self.block2_3(x2)  # x2 torch.Size([2, 128, 256, 256])
        x2 = self.block2_4(x2)  # x2 torch.Size([2, 128, 256, 256])

        e1 = self.conv_reduces[0](x1)  # e1 torch.Size([2, 64, 512, 512])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)  # e1 torch.Size([2, 64, 512, 512])
        e2 = self.conv_reduces[1](x2)  # e2 torch.Size([2, 64, 256, 256])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)  # e2 torch.Size([2, 64, 512, 512])

        out = torch.cat([e1, e2], dim=1)  # out torch.Size([2, 128, 512, 512])

        out = self.DSAM(out)  # out torch.Size([2, 64, 512, 512])

        return out
