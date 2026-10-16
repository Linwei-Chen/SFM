import re
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from numpy.lib.arraypad import pad
from numpy.lib.type_check import common_type
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List

from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.conv import Conv2d
import numpy as np



# CVPR 17' EDSR resblock
class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        )
        # self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)

# Panoptic FPN
# GropNorm Channel=16 per group or Groups=32 is the best


class ResConvReLUGN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=out_ch // 16)
        )

        self.residual_upsampler = torch.nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        )
        # self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class ConvReLUGN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=out_ch // 16)
        )

    def forward(self, input):
        return self.conv(input)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch=256, out_ch=1, activation=None):
        super().__init__()
        self.conv133 = nn.Sequential(
            BasicConv2d(in_channels=in_ch, out_channels=6 * in_ch // 32,
                        kernel_size=1, padding=0),
            BasicConv2d(in_channels=6 * in_ch // 32, out_channels=7 * in_ch // 32,
                        kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels=7 * in_ch // 32, out_channels=8 * in_ch // 32,
                        kernel_size=(3, 1), padding=(1, 0)),
        )
        self.conv15577 = nn.Sequential(
            BasicConv2d(in_channels=in_ch, out_channels=6 * in_ch // 32,
                        kernel_size=1, padding=0),
            BasicConv2d(in_channels=6 * in_ch // 32, out_channels=6 * in_ch // 32,
                        kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels=6 * in_ch // 32, out_channels=7 * in_ch // 32,
                        kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels=7 * in_ch // 32, out_channels=7 * in_ch // 32,
                        kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels=7 * in_ch // 32, out_channels=8 * in_ch // 32,
                        kernel_size=(1, 7), padding=(0, 3)),
        )
        self.conv13 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_ch, out_channels=4 * in_ch // 32,
                        kernel_size=1, padding=0),
        )
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels=in_ch, out_channels=12 * in_ch // 32,
                        kernel_size=1, padding=0),
        )

        self.final = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False)
        self.final_act = activation

    def forward(self, x):
        x1 = self.conv133(x)
        x2 = self.conv15577(x)
        x3 = self.conv13(x)
        x4 = self.conv1(x)
        res = self.final(torch.cat([x1, x2, x3, x4], dim=1))
        if self.final_act is not None:
            res = self.final_act(res)
        return res


def test_inceptionblock():
    x = torch.rand(2, 256, 4, 4)
    m = InceptionBlock(256, 1)
    y = m(x)
    print(y)
    print(y.shape)


class FPNDecoder(nn.Module):
    def __init__(self):
        super(FPNDecoder, self).__init__()
        ConvFunc = ConvReLUGN

        # warning! no padding??????
        self.decoders = nn.ModuleList([
            nn.Sequential(
                ConvFunc(in_ch=256, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True)
            ),
            nn.Sequential(
                ConvFunc(in_ch=256, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
            ),
            nn.Sequential(
                ConvFunc(in_ch=256, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                ConvFunc(in_ch=128, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
            ),
            nn.Sequential(
                ConvFunc(in_ch=256, out_ch=128),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
            ),
            nn.Sequential(
                ConvFunc(in_ch=256, out_ch=128),
            )
        ][::-1])
        self.final = nn.Conv2d(128, 3, kernel_size=1, bias=False)

    def forward(self, noisy_img, x):
        """
        x: fpn features
        """
        result = []
        for fpn_x, decoder in zip(x, self.decoders):
            temp_result = decoder(fpn_x)
            result.append(temp_result)
            # print(fpn_x.shape, temp_result.shape)
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0, keepdim=False)
        result = self.final(result)
        result = F.interpolate(
            result, size=noisy_img.shape[-2:], mode='bilinear', align_corners=True)
        return result
        


class CosLoss(torch.nn.Module):
    def __init__(self, subscale=1):
        super().__init__()
        self.subscale = subscale

    def forward(self, x, target):
        if self.subscale > 1:
            x = F.avg_pool2d(x, self.subscale)
            target = F.avg_pool2d(target, self.subscale)

        m_batchsize, C, height, width = x.size()
        # x = x.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # m_batchsize, C, height, width = target.size()
        # target = target.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # dot_mat = torch.bmm(x.permute(0,2,1), target).view(m_batchsize, 1, height, width)
        # dot_mat = torch.bmm(x.permute(0,2,1), target)
        # print(dot_mat.shape)
        x_dot_target = torch.sum(x * target, dim=1, keepdim=True)
        x2 = torch.norm(x, p=2, dim=1, keepdim=True)
        target2 = torch.norm(target, p=2, dim=1, keepdim=True)
        return x_dot_target / (x2 * target2)


def CORAL(source, target):
    source = F.adaptive_avg_pool2d(source, (1, 1)).squeeze()
    target = F.adaptive_avg_pool2d(target, (1, 1)).squeeze()
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(source.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(target.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    # loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


class FALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625, norm=1,):
        super(FALoss,self).__init__()
        self.subscale=int(1/subscale)
        self.norm = norm

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        L2norm=torch.norm(feature1, p=2, dim=1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        L2norm=torch.norm(feature2, p=2, dim=1, keepdim=True).repeat(1,C,1)
        # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[N,W*H,W*H]

        L1norm=torch.norm(mat2-mat1, self.norm)

        return L1norm/((height*width)**2) 


def unnormalize(data: torch.Tensor, std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5]):
    if data.dim() == 4:
        std = torch.tensor(std).reshape((1, 3, 1, 1)).to(data.device)
        mean = torch.tensor(mean).reshape((1, 3, 1, 1)).to(data.device)
    elif data.dim() == 3:
        std = torch.tensor(std).reshape((3, 1, 1)).to(data.device)
        mean = torch.tensor(mean).reshape((3, 1, 1)).to(data.device)
    else:
        raise NotImplementedError
    return data * std + mean


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


def test_non_local():
    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
    net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = torch.zeros(2, 3, 20, 20)
    net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = torch.randn(2, 3, 8, 20, 20)
    net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())


class DnCNN(nn.Module):
    def __init__(self, num_layers=9, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual



class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=3,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)

        return output


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        block.append(nn.Dropout2d(p=0.15)) # edited
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class ResizerBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.leakyrelu(out)
        return out


class enhancenet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.conv7x7=nn.Conv2d(3, 16, kernel_size=7, stride=1,padding=3, dilation=1, bias=False)
        self.conv1x1=nn.Conv2d(16, 16, kernel_size=1, stride=1,padding=0, dilation=1, bias=False)
        self.conv3x3=nn.Conv2d(16, 16, kernel_size=3, stride=1,padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        # self.bn1 = SynchronizedBatchNorm2d(16, momentum=bn_mom)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.resblocks = nn.Sequential(
            ResizerBasicBlock(16, 16),
            # ResizerBasicBlock(16, 16),
            # ResizerBasicBlock(16, 16)
        )
        self.finalconv7x7 = nn.Conv2d(16, 3, kernel_size=7, stride=1,padding=3, dilation=1, bias=False)
        
        # self.metasr = MetaSR(16, 16)
        
        

    def forward(self, x, size=None):
        if size is None:
            _, _, h, w = x.shape
        else:
            h, w = size
        out = self.conv7x7(x)
        out = self.act(out)
        
        out = self.conv1x1(out)
        out = self.act(out)
        out = self.bn1(out)
        
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        # out = self.metasr(out, (h, w))

        out = self.bn2(self.conv3x3(self.resblocks(out))) + out
        out = self.finalconv7x7(out)

        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
        return x + out
    
    def only_resize(self, x, size=None):
        if size is None:
            _, _, h, w = x.shape
        else:
            h, w = size
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        return x


def test_unet():
    x = torch.rand(2, 3, 128, 128)
    m = UNet()
    y = m(x)
    print(y.shape)
    pass

def test_resizer():
    import random
    model = enhancenet()
    h, w = 128, 128
    x = torch.rand(1, 3, h, w)
    scale = random.uniform(0.1, 4)
    h, w = scale*h, scale*w,
    h, w = int(h), int(w) 
    y = model(x, (h, w))
    print(y.shape)


class AdaDConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        if use_channel: assert in_channels >= 4
        self.kernel_size = kernel_size
        self.pad = (kernel_size-1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
            mid_channel = min((kernel_size ** 2 // 2), 4)
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * mid_channel , stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * mid_channel), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=groups * mid_channel, out_channels=groups * kernel_size ** 2, stride=1,
                        kernel_size=1, bias=False, padding=0, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
                # nn.Softmax(dim=1)
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = in_channels, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.kaiming_normal_(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        weight = self.weight_net(x) 
        _weight = weight
        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
        weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

        if self.use_channel:
            tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
            # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
            # print(weight.shape)
            weight = weight * tmp
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        # shape:  B x C x H // stride x W //stride x ksize x ksize
        pad_x = pad_x.unfold(2, self.kernel_size,self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)

        # channel_weight = self.channel_net(x).pow(2).log()
        # channel_weight = self.channel_net(x).sigmoid()
        # res = res * channel_weight
        return res

def test_adaconv():
    x = torch.rand(2, 16, 16, 16)
    m = AdaDConv(in_channels=16)
    y = m(x)
    print(y.shape)
    pass
    
if __name__ == '__main__':
    # test_inceptionblock()
    # test_discriminator
    # test_ispnet()
    # test_non_local()
    # test_resizer()
    test_adaconv()

    


