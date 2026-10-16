# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch_dct as dct
from mmseg.models.segmentors.AdaConv import LocalPixelRelationConv

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class PSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class PSPHeadWithDCT(PSPHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """
    def __init__(self, patch_size=8, compress_channel=0, use_lpr=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        if compress_channel <= 0:
            self.conv_seg = nn.Conv2d(self.channels, self.num_classes * self.patch_size ** 2, kernel_size=1)
        else:
            self.conv_seg = nn.Sequential(
                nn.Conv2d(self.channels, compress_channel, kernel_size=1),
                nn.SyncBatchNorm(compress_channel),
                nn.ReLU(True),
                nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
            )
        self.T = nn.parameter.Parameter(torch.Tensor([1.]))
        self.use_lpr = use_lpr
        if use_lpr:
            self.LPR = LocalPixelRelationConv(self.num_classes, kernel_size=3, stride=1, groups=1, dilation=patch_size)

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        output = nn.PixelShuffle(self.patch_size)(output)
        b, c, h, w = output.shape
        # print(output.shape)
        output = torch.nn.functional.unfold(output, (self.patch_size , self.patch_size), stride=self.patch_size) # c, c * 8 * 8, h // 8, w // 8
        output = output.reshape(b, self.num_classes, self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        output = output.permute(0, 1, 3, 4, 2)
        output = output.reshape(b, -1, self.patch_size, self.patch_size)
        # print(output.shape)
        # exit()
        # output = output.reshape(b, -1, 8, 8)
        output = dct.idct_2d(output) * 8 * self.T
        # output = output * self.T
        output = output.reshape(b, self.num_classes, h // self.patch_size, w // self.patch_size, self.patch_size * self.patch_size)
        output = output.permute(0, 1, 4, 2, 3)
        output = output.reshape(b, self.num_classes * self.patch_size * self.patch_size, h // self.patch_size * w // self.patch_size)
        output = torch.nn.functional.fold(output, (h, w), (self.patch_size, self.patch_size), stride=self.patch_size)
        if self.use_lpr:
            pixel_relation, output = self.LPR(output, output, use_checkpoint=True)
            # print('lpr')
        # print(output.shape)
        # output = output.reshape(b, c, h, w)
        # print(output.shape)
        # print(output)
        return output