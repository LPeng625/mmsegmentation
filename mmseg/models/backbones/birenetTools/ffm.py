# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

import math


class STDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels before scaling.
        stride (int): The number of stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2 ** (i + 1) if i != num_convs - 1 else 2 ** i
            self.layers.append(
                ConvModule(
                    out_channels // 2 ** i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        x_out = x.view(x.size(0), -1)
        return x_out


class ChannelGate(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = channel

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        scale = scale.unsqueeze(2)
        scale = scale.unsqueeze(3)
        scale = scale.expand_as(x)
        return x * scale


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
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


# TODO CSAM 通道空间注意力
class CSAM(nn.Module):
    def __init__(self, channel):
        super(CSAM, self).__init__()
        self.ChannelGate = ECAAPMP(channel)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# TODO CSAM 通道空间注意力（ECAAP）
class CSAM_nolyAP(nn.Module):
    def __init__(self, channel):
        super(CSAM_nolyAP, self).__init__()
        self.ChannelGate = ECAAP(channel)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# TODO CSAM 通道空间注意力（ECAMP）
class CSAM_nolyMP(nn.Module):
    def __init__(self, channel):
        super(CSAM_nolyMP, self).__init__()
        self.ChannelGate = ECAMP(channel)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


# TODO CBAM 通道空间注意力
class CBAM(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(channel, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


# TODO SELayer 通道注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# TODO ECA 原版，只有平均池化
class ECAAP(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECAAP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 控制根据输入的channel控制kernel_size 的大小，保证kernel_size是一个奇数
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

# TODO ECA 改进版，只有最大池化
class ECAMP(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECAMP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 控制根据输入的channel控制kernel_size 的大小，保证kernel_size是一个奇数
        padding = kernel_size // 2
        self.amp_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.amp_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

# TODO ECA 改进版，有平均池化和最大池化
class ECAAPMP(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAPMP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.amp_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        amp = self.amp_pool(x).view([b, 1, c])
        out = torch.cat([avg, amp], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # TODO 注意力机制消融，CSAM 为提出的通道空间注意力
        self.FFM_has_CSAM = True
        self.FFM_has_CSAM_onlyAP = False
        self.FFM_has_CSAM_onlyMP = False
        self.FFM_has_CBAM = False
        self.FFM_has_SE = False
        self.FFM_has_SAM = False
        self.FFM_has_ECA = False

        self.init_conv = ConvModule(
            2 * out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if self.FFM_has_CSAM:
            self.atten = CSAM(out_channels)

        if self.FFM_has_CSAM_onlyAP:
            self.atten = CSAM_nolyAP(out_channels)

        if self.FFM_has_CSAM_onlyMP:
            self.atten = CSAM_nolyMP(out_channels)

        if self.FFM_has_CBAM:
            self.atten = CBAM(out_channels)

        if self.FFM_has_SE:
            self.atten = SELayer(out_channels)

        if self.FFM_has_SAM:
            self.atten = SpatialGate()

        if self.FFM_has_ECA:
            self.atten = ECAAP(out_channels)

    def forward(self, spatial_inputs, context_inputs):

        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.init_conv(inputs)
        x_atten = self.atten(x)
        return x_atten + x
        # return x




class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding='same', stride=1):
        super().__init__()
        # 自动计算 padding 以保持空间尺寸（当 padding='same' 时）
        if isinstance(padding, str) and padding.lower() == 'same':
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,  # 任意 (n, m) 卷积核
            padding=padding,  # 自定义填充
            stride=stride,  # 默认步长为1
            bias=False  # BN层不需要卷积偏置
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# TODO SAM CBAM空间注意力部分
class SpatialGate_new(nn.Module):
    def __init__(self):
        super(SpatialGate_new, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        self.conv_1x3 = CustomConv2d(2, 1, kernel_size=(1, 3))
        self.conv_3x1 = CustomConv2d(1, 1, kernel_size=(3, 1))
        self.conv_1x5 = CustomConv2d(2, 1, kernel_size=(1, 5))
        self.conv_5x1 = CustomConv2d(1, 1, kernel_size=(5, 1))
        self.conv_1x7 = CustomConv2d(2, 1, kernel_size=(1, 7))
        self.conv_7x1 = CustomConv2d(1, 1, kernel_size=(7, 1))

    def forward(self, x):
        x_compress = self.compress(x)

        x_1 = self.spatial(x_compress)
        x_3 = self.conv_3x1(self.conv_1x3(x_compress))
        x_5 = self.conv_5x1(self.conv_1x5(x_compress))
        x_7 = self.conv_7x1(self.conv_1x7(x_compress))
        x_out = x_1 + x_3 + x_5 + x_7

        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

# TODO CSAM 通道空间注意力
class CSAM_new2(nn.Module):
    def __init__(self, channel):
        super(CSAM_new2, self).__init__()
        self.ChannelGate = ECAAPMP(channel)
        self.SpatialGate = SpatialGate_new()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class FeatureFusionModule_new2(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # TODO 注意力机制消融，CSAM 为提出的通道空间注意力
        self.FFM_has_CSAM_new2 = True
        self.FFM_has_CSAM_onlyAP = False
        self.FFM_has_CSAM_onlyMP = False
        self.FFM_has_CBAM = False
        self.FFM_has_SE = False
        self.FFM_has_SAM = False
        self.FFM_has_ECA = False

        self.init_conv = ConvModule(
            2 * out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if self.FFM_has_CSAM_new2:
            self.atten = CSAM_new2(out_channels)

        if self.FFM_has_CSAM_onlyAP:
            self.atten = CSAM_nolyAP(out_channels)

        if self.FFM_has_CSAM_onlyMP:
            self.atten = CSAM_nolyMP(out_channels)

        if self.FFM_has_CBAM:
            self.atten = CBAM(out_channels)

        if self.FFM_has_SE:
            self.atten = SELayer(out_channels)

        if self.FFM_has_SAM:
            self.atten = SpatialGate()

        if self.FFM_has_ECA:
            self.atten = ECAAP(out_channels)


    def forward(self, spatial_inputs, context_inputs):

        x = torch.cat([spatial_inputs, context_inputs], dim=1)

        x = self.init_conv(x)

        x_atten = self.atten(x)

        return x + x_atten
