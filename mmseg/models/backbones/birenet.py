"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial
from .birenetTools import edm

from .birenetTools.ffm import FeatureFusionModule, FeatureFusionModule_new2

nonlinearity = partial(F.relu, inplace=True)
from mmseg.registry import MODELS

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)  # x torch.Size([2, 512, 32, 32]) -> torch.Size([2, 128, 32, 32])
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)  # x torch.Size([2, 128, 32, 32]) -> torch.Size([2, 128, 64, 64])
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)  # x torch.Size([2, 128, 64, 64]) -> torch.Size([2, 256, 64, 64])
        x = self.norm3(x)
        x = self.relu3(x)
        return x

@MODELS.register_module()
class BiReNet34(nn.Module):
    def __init__(self, out_channels=2):
        super(BiReNet34, self).__init__()
        # TODO 模块消融和训练策略选择
        self.is_Train = True
        self.has_DB = False
        self.has_FFM = True
        self.has_EDM = True
        self.has_AuxHead = True

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        resnet.load_state_dict(torch.load('pre_trained_weights/resnet34-b627a593.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        if self.has_DB:
            self.dblock = Dblock(512)

        if self.has_FFM:
            self.FeatureFusionModule1 = FeatureFusionModule(
                filters[0],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

            self.FeatureFusionModule2 = FeatureFusionModule(
                filters[1],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

            self.FeatureFusionModule3 = FeatureFusionModule(
                filters[2],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

        if self.has_EDM:
            self.EDM = edm.pidinet_init('carv4', init_stride=1, is_ori=True, inplane=64)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        # EDM h w 3 -> h/2 w/2 64
        if self.has_EDM:
            edge = self.EDM(x)  # edge torch.Size([2, 64, 512, 512])

        # Init h w 3 -> h/2 w/2 64
        x = self.firstconv(x)  # x torch.Size([2, 3, 1024, 1024])->torch.Size([2, 64, 512, 512])
        x = self.firstbn(x)
        x = self.firstrelu(x)

        # Encoder
        # h/2 w/2 64 -> h/32 w/32 512
        e1 = self.firstmaxpool(x)  # x torch.Size([2, 64, 256, 256]) e1 torch.Size([2, 64, 256, 256])
        e1 = self.encoder1(e1)  # e1 torch.Size([2, 64, 256, 256]) -> torch.Size([2, 64, 256, 256])
        e2 = self.encoder2(e1)  # e1 torch.Size([2, 64, 256, 256]) -> e2 torch.Size([2, 128, 128, 128])
        e3 = self.encoder3(e2)  # e2 torch.Size([2, 128, 128, 128]) -> e3 torch.Size([2, 256, 64, 64])
        e4 = self.encoder4(e3)  # e3 torch.Size([2, 256, 64, 64]) -> e4 torch.Size([2, 512, 32, 32])

        if self.has_DB:
            e4 = self.dblock(e4)

        # Decoder
        if self.has_FFM:
            d4 = self.FeatureFusionModule3(self.decoder4(e4), e3)  # d4 torch.Size([2, 256, 64, 64])
            d3 = self.FeatureFusionModule2(self.decoder3(d4), e2)  # d3 torch.Size([2, 128, 128, 128])
            d2 = self.FeatureFusionModule1(self.decoder2(d3), e1)  # d2 torch.Size([2, 64, 256, 256])
        else:
            d4 = self.decoder4(e4) + e3
            d3 = self.decoder3(d4) + e2
            d2 = self.decoder2(d3) + e1

        # 将EDM融合到主分支
        if self.has_EDM:
            d1 = self.decoder1(d2) + edge  # d1 torch.Size([2, 64, 512, 512])
        else:
            d1 = self.decoder1(d2)

        out = self.finalconv3(self.finalrelu2(self.finalconv2(self.finalrelu1(self.finaldeconv1(
            d1)))))  # self.finaldeconv1(d1) torch.Size([2, 32, 1024, 1024]) out torch.Size([2, 1, 1024, 1024])
        # out = F.sigmoid(out)

        if self.is_Train:
            # TODO 采用增强训练策略，即给EDM分支添加辅助训练头
            if self.has_EDM and self.has_AuxHead:
                out_e = self.finalconv3(self.finalrelu2(self.finalconv2(self.finalrelu1(self.finaldeconv1(edge)))))
                # out_e = F.sigmoid(out_e)
                outs = [out] + [out_e]
                outs = [outs[i] for i in (0, 1)]
            else:
                outs = [out]
                outs = [outs[i] for i in (0,)]
            return tuple(outs)
        else:
            return out


class BiReNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(BiReNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class BiReNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(BiReNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class BiReNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(BiReNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


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


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, q, k, v):
        # [N, C, H , W]
        b, c, h, w = q.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(q).reshape(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(k).reshape(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(v).reshape(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + q + v
        return out
