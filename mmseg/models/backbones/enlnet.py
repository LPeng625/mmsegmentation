import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

###############################################################################
# Helper Functions
###############################################################################
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

from mmseg.registry import MODELS


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
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SemanticTokensNonLocalBlock(nn.Module):
    def __init__(self, in_channels, size=(32, 32)):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channel = self.in_channels // 2
        self.conv_g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.in_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.pooling_size = 2
        self.token_len = self.pooling_size * self.pooling_size

        self.to_qk = nn.Linear(self.in_channels, 2 * self.inter_channel, bias=False)


        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=False),
            nn.Sigmoid()
        )

        self.with_pos = False
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, 4, in_channels))

        self.with_d_pos = False
        if self.with_d_pos:
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.inter_channel,
                                                                  size[0], size[1]))


    def _forward_semantic_tokens_channel_attention(self, x):
        b, c, h, w = x.shape  # torch.Size([2, 32, 64, 64])
        spatial_attention = self.conv_a(x)  # torch.Size([2, 4, 64, 64])
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()  # torch.Size([2, 4, 4096])
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  # torch.Size([2, 32, 4096])

        # TODO add channel attention
        channel_attention = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(channel_attention).view(b, c, 1)
        x = x * channel_attention

        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)  # 2, 4, 4096 / 2, 32, 4096 / 2, 4, 32

        return tokens

    def forward(self, x): # 2 256 64 64

        # [N, C, H , W]
        b, c, h, w = x.size()

        x_clone = x

        x = self._forward_semantic_tokens_channel_attention(x)

        if self.with_pos:
            x = x + self.pos_embedding  # 2, 4, 32

        _, n, _ = x.size()
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = qk[0].reshape(b, -1, n), qk[1] # 2, 4, 16  / 2, 16, 4

        if self.with_d_pos:
            x_g = (self.conv_g(x_clone) + self.pos_embedding_decoder).reshape(b, c // 2, -1).permute(0, 2,
                                                                                                     1).contiguous()  # 2 64 * 64 16
        else:
            x_g = self.conv_g(x_clone).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()  # 2 64 * 64 16

        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(q, k) # 2 16 16

        mul_theta_phi = self.softmax(mul_theta_phi) # 2 16 16
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(x_g, mul_theta_phi) # 2 64 * 64 16
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)

        out = mask + x_clone

        return out

@MODELS.register_module()
class ENLNet(nn.Module):

    def __init__(self, out_channels=1):
        super(ENLNet, self).__init__()

        # resnet
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('pre_trained_weights/resnet34-b627a593.pth'))
        self.resnet = resnet
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # TODO add encoder
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # TODO add decoder
        dims_decoder = [512, 256, 128, 64]
        self.decoder4 = DecoderBlock(dims_decoder[0], dims_decoder[1])
        self.decoder3 = DecoderBlock(dims_decoder[1], dims_decoder[2])
        self.decoder2 = DecoderBlock(dims_decoder[2], dims_decoder[3])
        self.decoder1 = DecoderBlock(dims_decoder[3], dims_decoder[3])
        self.finaldeconv1 = nn.ConvTranspose2d(dims_decoder[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_channels, 3, padding=1)


        # TODO 5layer
        self.conv_pred_5layer = nn.Conv2d(512, 32, kernel_size=3, padding=1)
        self.conv_pred_back_5layer = nn.Conv2d(32, 512, kernel_size=3, padding=1)

        # TODO LoveDA DeepGlobe SW_GF2_1024 64  SW_GF2_512 32 CVC 16
        self.SemanticTokensNonLocalBlock_hw32 = SemanticTokensNonLocalBlock(in_channels=32, size=(32, 32))


    def forward_features_linknet_5layer(self, x):

        skip_list = []

        # TODO birenet init
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)


        e1_l = self.encoder1(x)
        skip_list.append(e1_l)

        e2_l = self.encoder2(e1_l)
        skip_list.append(e2_l)

        e3_l = self.encoder3(e2_l)
        skip_list.append(e3_l)

        e4_l = self.encoder4(e3_l)
        skip_list.append(e4_l)

        return e4_l, skip_list


    def SemanticTokensNonLocal_block_5layer(self, x1):
        # 降维
        x1 = self.conv_pred_5layer(x1)
        #  forward tokenzier
        x1 = self.SemanticTokensNonLocalBlock_hw32(x1)
        # feature differencing
        # 恢复维度
        x1 = self.conv_pred_back_5layer(x1)
        return x1

    def up_features_linknet_Center_SemanticTokensNonLocal_5layer(self, x, skip_list):

        # TODO 交叉注意力
        x = x + self.SemanticTokensNonLocal_block_5layer(x)

        d4 = self.decoder4(x) + skip_list[2]


        d3 = self.decoder3(d4) + skip_list[1]


        d2 = self.decoder2(d3) + skip_list[0]


        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


    def forward(self, x1):

        # TODO encoder
        x1, skip_list = self.forward_features_linknet_5layer(x1)

        # TODO decoder
        x = self.up_features_linknet_Center_SemanticTokensNonLocal_5layer(x1, skip_list)

        return self.sigmoid(x)
