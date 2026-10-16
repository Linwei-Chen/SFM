# Copyright (c) OpenMMLab. All rights reserved.
from operator import index
import sys
from turtle import forward

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import warnings

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models.backbones.interp2d import SaliencySampler, SoftPool2d
from torch.utils.checkpoint import checkpoint
from mmseg.models.segmentors.AdaConv import LocalPixelRelationConv
# from interp2d import test_2
# test_2()
from torch.utils.tensorboard import SummaryWriter
import copy
from ..builder import BACKBONES
from ..utils import ResLayer
import torch_dct as dct

# from mmseg.models.builder import BACKBONES
# from mmseg.models.utils import ResLayer


class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        # assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        
        self.with_dcn = dcn is not None
        if self.with_dcn:
            dcn_copy = dcn.copy()
            fallback_on_stride = dcn_copy.pop('fallback_on_stride', False)
            only_on_stride = dcn_copy.pop('only_on_stride', False)
            only_on_stride_conv1 = dcn_copy.pop('only_on_stride_conv1', False)
            # print(only_on_stride, self.conv2_stride==1)
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv1 = build_conv_layer(
                dcn_copy,
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
            self.add_module(self.norm1_name, norm1)
            self.conv2 = build_conv_layer(
                dcn_copy,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False)
            self.add_module(self.norm2_name, norm2)
        else:
            self.conv1 = build_conv_layer(
                conv_cfg,
                inplanes,
                planes,
                3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
            self.add_module(self.norm1_name, norm1)
            self.conv2 = build_conv_layer(
                conv_cfg, planes, planes, 3, padding=1, bias=False)
            self.add_module(self.norm2_name, norm2)
            fallback_on_stride = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is
    "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                #  index=1, # 20220913
                **kwargs,
                 ):
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        fallback_on_stride = False
        if self.with_dcn:
            dcn_copy = dcn.copy()
            fallback_on_stride = dcn_copy.pop('fallback_on_stride', False)
            only_on_stride = dcn_copy.pop('only_on_stride', False)
            only_on_stride_conv1 = dcn_copy.pop('only_on_stride_conv1', False)
            # print(only_on_stride, self.conv2_stride==1)
        if self.with_dcn and only_on_stride_conv1 and (self.conv1_stride==2 or self.conv2_stride==2):
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv1 = build_conv_layer(
                dcn_copy,
                inplanes,
                planes,
                kernel_size=1,
                stride=self.conv1_stride,
                padding=0,
                dilation=1,
                bias=False)
        else:
            self.conv1 = build_conv_layer(
                conv_cfg,
                inplanes,
                planes,
                kernel_size=1,
                stride=self.conv1_stride,
                bias=False)
        self.add_module(self.norm1_name, norm1)

        if not self.with_dcn or fallback_on_stride or (only_on_stride and self.conv2_stride==1) or only_on_stride_conv1:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn_copy,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    This backbone is the improved implementation of `Deep Residual Learning
    for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            When dcn is not None, conv_cfg must be None. Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. The length of stage_with_dcn is equal to num_stages.
            Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'.
            Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmseg.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None, 
                 use_checkpoint=False,
                 **kargs):
        super(ResNet, self).__init__(init_cfg)
        self.use_checkpoint = use_checkpoint
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')
        self.block_init_cfg = block_init_cfg ###
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        inplanes = self.inplanes
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            inplanes = planes * self.block.expansion
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            if self.use_checkpoint: x.requires_grad = True
            x = checkpoint(self.stem, x) if self.use_checkpoint else self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if self.use_checkpoint:
                x = checkpoint(res_layer, x)
            else:
                x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1c(ResNet):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    """ResNetV1d variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

def conv_identify(weight, bias):
    weight.data.zero_()
    if bias is not None:
        bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, :, :] = 1.0

import numpy as np
class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat(self.channels,1,1,1))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Downsample_PASA_group_softmax(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=8):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]

class AdaFreq_Pooling(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, in_channels, groups=8):
        super().__init__()
        self.groups = groups
        self.freq_att = nn.Conv2d(in_channels=in_channels, out_channels=groups, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(groups)
        self.freq_att_bn = nn.BatchNorm2d(groups)
        self.freq_bias_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            # nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=groups, kernel_size=1, bias=False),
        )

        self.freq_att_bn.weight.data.zero_()
        self.freq_bias_conv[-1].weight.data.zero_()
        # self.freq_bias_conv[2].bias.data.zero_()


    def ferq_index_bias(self, x):
        n, _, out_h, out_w = x.shape
        # norm = torch.tensor([[[[out_w, out_h]]]]).type_as(x).to(x.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        r = w ** 2 + h ** 2 + 1e-8
        r_sq = r.sqrt()
        # grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2), 
                        #   r.unsqueeze(2), 
                          r_sq.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid.permute(0, 3, 1, 2) # b, 3, h, w
        return self.freq_bias_conv(grid)

    def forward(self, x):
        # if self.downsample:
        #     low_part = torch.fft.fftshift(torch.fft.fft2(x))[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        #     return torch.fft.ifft2(torch.fft.ifftshift(low_part)).real
        # else:
        #     low_part = torch.fft.fftshift(torch.fft.fft2(x))
        #     mask = torch.zeros_like(low_part, device=low_part.device)
        #     mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
        #     return torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
        # freq_att = self.freq_att_bn(self.freq_att(x_fft.abs())).sigmoid()
        
        att_bias = self.ferq_index_bias(x)
        # att = self.bn(att + att_bias).sigmoid()
        att = self.bn(att_bias).sigmoid()
        b, c, h, w = x_fft.shape
        x_fft = x_fft.reshape(b, self.groups, c // self.groups, h, w)
        att = att.reshape(b, self.groups, 1, h, w)
        # freq_att = freq_att.reshape(b, self.groups, 1, h, w)
        x_fft = x_fft * att
        x_fft = x_fft.reshape(b, c, h, w)
        return torch.fft.ifft2(torch.fft.ifftshift(x_fft)).real
    
def calculate_low_frequency_energy_ratio(tensor, index=None):
    # 将输入tensor转换为频域
    freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
    # 将低频部分移动到频域中心
    shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
    # 计算频域中的能量
    freq_domain_energy = torch.abs(shifted_freq_domain_tensor) ** 2
    freq_domain_energy = freq_domain_energy.mean(dim=0, keepdims=True)
    # print(freq_domain_energy)
    # 计算低频成分的能量
    b, c, h, w = freq_domain_energy.shape
    freq_thres = 1.41 * 0.25
    # freq_thres = 1 * 0.25
    mask = torch.zeros_like(freq_domain_energy, device=freq_domain_energy.device)
    mask[:,:,round(h/2 - h * freq_thres):round(h/2 + h * freq_thres), round(w/2 - w * freq_thres):round(w/2 + w * freq_thres)] = 1.0
    low_freq_energy = (mask * freq_domain_energy).sum((-3, -2, -1)) 
    # low_freq_energy = freq_domain_energy[..., h//2 - h//4:h//2 + h//4, w//2 - w//4:w//2 + w//4].sum((-3, -2, -1)) 
    # print(low_freq_energy)
    # 计算低频成分能量与整个频域能量的比值
    total_energy = freq_domain_energy.sum((-3, -2, -1))  # 计算整个频域的能量
    low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
    return 1. - low_freq_energy_ratio
    
@BACKBONES.register_module()
class ResNetV1cWithBlur(ResNetV1c):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, blur_k=3, 
                 blur_type='blur',
                 freq_thres=0.25,
                 log_aliasing_ratio=False,
                 **kwargs):
        # super().__init__(strides=(1, 1, 1, 1), **kwargs)
        super().__init__(**kwargs)
        if blur_type == 'blur':
            BLUR1 = Downsample(pad_type='reflect', filt_size=blur_k, stride=1, channels=256, pad_off=0)
            BLUR2 = Downsample(pad_type='reflect', filt_size=blur_k, stride=1, channels=512, pad_off=0)
            BLUR3 = Downsample(pad_type='reflect', filt_size=blur_k, stride=1, channels=1024, pad_off=0)
        elif blur_type == 'adablur':
            BLUR1 = Downsample_PASA_group_softmax(in_channels=256, kernel_size=3, stride=1, group=8)
            BLUR2 = Downsample_PASA_group_softmax(in_channels=512, kernel_size=3, stride=1, group=8)
            BLUR3 = Downsample_PASA_group_softmax(in_channels=1024, kernel_size=3, stride=1, group=8)
        elif blur_type == 'flc':
            BLUR1 = FLC_Pooling(downsample=False, freq_thres=freq_thres)
            BLUR2 = FLC_Pooling(downsample=False, freq_thres=freq_thres)
            BLUR3 = FLC_Pooling(downsample=False, freq_thres=freq_thres)
        elif blur_type == 'adafreq':
            BLUR1 = AdaFreq_Pooling(in_channels=256, groups=8)
            BLUR2 = AdaFreq_Pooling(in_channels=512, groups=8)
            BLUR3 = AdaFreq_Pooling(in_channels=1024, groups=8)
        elif blur_type == 'none':
            BLUR1 = nn.Identity()
            BLUR2 = nn.Identity()
            BLUR3 = nn.Identity()
        else:
            raise NotImplementedError
        self.AdaD = nn.ModuleList([
            nn.Sequential(
                # AdaDConv(in_channels=64, kernel_size=3, stride=2, groups=64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            BLUR1,
            BLUR2,
            BLUR3,
        ])
        self.aliasing_ratio=[[] for i in range(6)]
        self.log_aliasing_ratio=log_aliasing_ratio

    def forward(self, x, denoise=True):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = self.AdaD[i](x)
            x = res_layer(x)
            if denoise:
                pass
            if i in self.out_indices:
                # print(i)
                outs.append(x)
            if i != len(self.res_layers) - 1:
                pass
        if self.log_aliasing_ratio:
            self.aliasing_ratio[0].append(calculate_low_frequency_energy_ratio(outs[0]))
            self.aliasing_ratio[1].append(calculate_low_frequency_energy_ratio(outs[1]))
            self.aliasing_ratio[2].append(calculate_low_frequency_energy_ratio(outs[2]))
            self.aliasing_ratio[3].append(calculate_low_frequency_energy_ratio(self.AdaD[1](outs[0])))
            self.aliasing_ratio[4].append(calculate_low_frequency_energy_ratio(self.AdaD[2](outs[1])))
            self.aliasing_ratio[5].append(calculate_low_frequency_energy_ratio(self.AdaD[3](outs[2])))
            if (len(self.aliasing_ratio[0]) % 10) == 0:
                print(sum(self.aliasing_ratio[0]) / len( self.aliasing_ratio[0]))
                print(sum(self.aliasing_ratio[1]) / len( self.aliasing_ratio[1]))
                print(sum(self.aliasing_ratio[2]) / len( self.aliasing_ratio[2]))
                print(sum(self.aliasing_ratio[3]) / len( self.aliasing_ratio[3]))
                print(sum(self.aliasing_ratio[4]) / len( self.aliasing_ratio[4]))
                print(sum(self.aliasing_ratio[5]) / len( self.aliasing_ratio[5]))
        return tuple(outs)
    
class FLC_Pooling(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, downsample=False, freq_thres=0.25):
        super(FLC_Pooling, self).__init__()
        assert freq_thres < (0.5 + 1e-8)
        assert freq_thres > 0.0
        self.downsample = downsample
        self.freq_thres = freq_thres

    def forward(self, x):
        if self.downsample:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
            return torch.fft.ifft2(torch.fft.ifftshift(low_part)).real
        else:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))
            mask = torch.zeros_like(low_part, device=low_part.device)
            # mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
            _, _, h, w = x.shape
            mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
            return torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real


# @MODELS.register_module()
class ResNetV1d(ResNet):
    """ResNetV1d variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)

class AttBottleneck(Bottleneck):
    def __init__(self, 
                att=nn.Identity,
                att_cfg={},
                att_pos='after_bn3',
                train_first_stage=True, 
                train_last_stage=True, 
                **kwargs):
        super().__init__(**kwargs)
        self.att_pos = att_pos
        self.train_first_stage = train_first_stage
        self.train_last_stage = train_last_stage
        if (self.planes == 64 and not self.train_first_stage) or (self.planes == 512 and not self.train_last_stage):
            self.att = nn.Identity()
        else:
            if self.att_pos in ('after_bn3', 'after_conv3'):
                self.att = att(in_channels=self.planes * self.expansion, **att_cfg)
            elif self.att_pos in ('after_bn2', 'after_conv2', 'after_bn1', 'after_conv1', 'after_relu1'):
                self.att = att(in_channels=self.planes, **att_cfg)
            else:
                raise NotImplementedError

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.att_pos == 'after_relu1': out = self.att(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            if self.att_pos == 'after_conv2': out = self.att(out)
            out = self.norm2(out)
            if self.att_pos == 'after_bn2': out = self.att(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            if self.att_pos == 'after_conv3': out = self.att(out)
            out = self.norm3(out)
            if self.att_pos == 'after_bn3': out = self.att(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

import mmcv
from mmcv.cnn import ConvModule
class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                        #   dict(type='HSigmoid', bias=3.0, divisor=6.0),
                          dict(type='Sigmoid')
                          )
                          ):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            # out_channels=make_divisible(channels // ratio, 8),
            out_channels=channels // ratio,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            # in_channels=make_divisible(channels // ratio, 8),
            in_channels=channels // ratio,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class SEBottleneck(AttBottleneck):
    def __init__(self, 
                att=SELayer,
                att_cfg={'reduction':16},
                train_first_stage=False, **kwargs):
        super().__init__(att=att, att_cfg=att_cfg, train_first_stage=train_first_stage, **kwargs)


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # return x * y.expand_as(x)
        return y.expand_as(x)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, out=1):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, out, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting         
        return scale

class FrequencyMix(nn.Module):
    def __init__(self, 
                in_channels,
                k_list=[2],
                # freq_list=[2, 3, 5, 7, 9, 11],
                fs_feat='feat',
                lp_type='freq_channel_att',
                act='sigmoid',
                channel_res=True,
                spatial='conv',
                spatial_group=1,
                ):
        super().__init__()
        k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        self.channel_res = channel_res
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        if spatial == 'conv':
            self.freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=(len(k_list) + 1) * self.spatial_group, 
                                            stride=1,
                                            kernel_size=3, padding=1, bias=True) 
            self.freq_weight_conv.weight.data.zero_()
            self.freq_weight_conv.bias.data.zero_()   
        elif spatial == 'cbam': 
            self.freq_weight_conv = SpatialGate(out=len(k_list) + 1)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReflectionPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type in ('freq_channel_att', 'freq_channel_att_reduce_high'):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            # self.channel_att[1].weight.data.zero_()
            # self.channel_att[1].bias.data.zero_()
            # self.channel_att.weight.data.zero_()
        elif self.lp_type in ('freq_eca', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.ModuleList(
                [eca_layer(self.in_channels, k_size=9) for _ in range(len(k_list) + 1)]
            )
        elif self.lp_type in ('freq_channel_se', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = SELayer(self.in_channels, ratio=16)
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))


    def forward(self, x):
        freq_weight = self.freq_weight_conv(x)
        
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        
        x_list = []
        if self.lp_type == 'avgpool':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(freq_weight[:, idx:idx+1] * high_part)
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1])
        elif self.lp_type == 'freq':
            pre_x = x
            b, _c, h, w = freq_weight.shape
            freq_weight = freq_weight.reshape(b, self.spatial_group, -1, h, w)
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part
                tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        elif self.lp_type in ('freq_channel_att', 'freq_eca', 'freq_channel_se'):
            pre_freq = 1
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            h, w =  int(h), int(w) 
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                channel_att_mask = mask.clone()
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part
                # print('hw:', h, w)
                # print(idx, 'int:', freq, int(h/2 - h/(2 * pre_freq)), int(h/2 + h/(2 * pre_freq)), int(w/2 - w/(2 * pre_freq)), int(w/2 + w/(2 * pre_freq)))
                # print(idx, 'int:', freq, int(h/2 - h/(2 * freq)), int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)), int(w/2 + w/(2 * freq)))
                # print('hw:', h, w)
                # print(idx, ':', freq, round(h/2 - h/(2 * pre_freq)), round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)), round(w/2 + w/(2 * pre_freq)))
                # print(idx, ':', freq, round(h/2 - h/(2 * freq)), round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)), round(w/2 + w/(2 * freq)))
                channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                channel_att_mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                # pre_freq = int(freq)
                pre_freq = freq
                if isinstance(self.channel_att, nn.ModuleList):
                    c_att = self.channel_att[idx](((x_fft * channel_att_mask).abs() + 1).log())
                else:
                    c_att = self.channel_att(((x_fft * channel_att_mask).abs() + 1).log())
                    # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
                c_att = (c_att + 0.5) if self.channel_res else (c_att * 2.0)
                x_list.append(freq_weight[:, idx:idx+1] * high_part * c_att)

            channel_att_mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            if isinstance(self.channel_att, nn.ModuleList):
                c_att = self.channel_att[len(x_list)](((x_fft * channel_att_mask).abs() + 1).log())
            else:
                c_att = self.channel_att(((x_fft * channel_att_mask).abs() + 1).log())
                # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
            c_att = (c_att + 0.5) if self.channel_res else (c_att * 2.0)
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1] * c_att)
        elif self.lp_type == 'freq_channel_att_reduce_high':
            pre_freq = 1
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            h, w =  int(h), int(w) 
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                channel_att_mask = mask.clone()
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part

                channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                channel_att_mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                pre_freq = int(freq)
                # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
                c_att = self.channel_att((x_fft * channel_att_mask).abs())
                x_list.append((1. - freq_weight[:, idx:idx+1]) * high_part * (1. - c_att))
            channel_att_mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
            c_att = self.channel_att((x_fft * channel_att_mask).abs())
            x_list.append(pre_x * (freq_weight[:, len(x_list):len(x_list)+1] + 1) * (c_att + 1))
        x = sum(x_list)
        return x


class _FrequencyMix(nn.Module):
    '''
    Frequency Mixing module
    '''
    def __init__(self, 
                in_channels,
                k_list=[2],
                # freq_list=[2, 3, 5, 7, 9, 11],
                fs_feat='feat',
                lp_type='freq_channel_att',
                act='sigmoid',
                channel_res=True,
                spatial='conv',
                spatial_group=1,
                compress_ratio=16,
                ):
        super().__init__()
        k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        self.channel_res = channel_res
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        if spatial == 'conv':
            self.freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=(len(k_list) + 1) * self.spatial_group, 
                                            stride=1,
                                            kernel_size=3, padding=1, bias=True) 
            # self.freq_weight_conv.weight.data.zero_()
            # self.freq_weight_conv.bias.data.zero_()   
        elif spatial == 'cbam': 
            self.freq_weight_conv = SpatialGate(out=len(k_list) + 1)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReflectionPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type in ('freq_channel_att', 'freq_channel_att_reduce_high'):
            # self.channel_att= nn.ModuleList()
            # for i in 
            self.channel_att_low = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            # self.channel_att_low[3].weight.data.zero_()
            # self.channel_att_low[3].bias.data.zero_()

            self.channel_att_high = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            # self.channel_att_high[3].weight.data.zero_()
            # self.channel_att_high[3].bias.data.zero_()
            # self.channel_att.weight.data.zero_()
        elif self.lp_type in ('freq_eca', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.ModuleList(
                [eca_layer(self.in_channels, k_size=9) for _ in range(len(k_list) + 1)]
            )
        elif self.lp_type in ('freq_channel_se', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = SELayer(self.in_channels, ratio=16)
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.freq_thres=0.25 * 1.4

    def forward(self, x):
        freq_weight = self.freq_weight_conv(x)
        
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid()
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
        low_mask = torch.zeros_like(x_fft, device=x_fft.device)
        high_mask = torch.ones_like(x_fft, device=x_fft.device)
        # mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
        _, _, h, w = x.shape
        low_mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
        high_mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 0.0

        low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * low_mask)).real
        high_part = x - low_part
        low_x_fft = x_fft * low_mask
        high_x_fft = x_fft * high_mask
        low_c_att = torch.sqrt(self.channel_att_low(low_x_fft.real) ** 2 + self.channel_att_low(low_x_fft.imag) ** 2 + 1e-8)
        high_c_att = torch.sqrt(self.channel_att_high(high_x_fft.real) ** 2 + self.channel_att_high(high_x_fft.imag) ** 2 + 1e-8)
        low_part = low_part * freq_weight[:, 0:1,] * low_c_att
        high_part = high_part * freq_weight[:, 1:2,] * high_c_att
        # low_part = low_part * freq_weight[:, 0:1,] * self.channel_att_low((x_fft * low_mask).abs()) 
        # high_part = high_part * freq_weight[:, 1:2,] * self.channel_att_high((x_fft * high_mask).abs())
        res = low_part + high_part
        if self.channel_res: res += x
        return res

class NyBottleneck(AttBottleneck):
    def __init__(self, 
                att=_FrequencyMix,
                att_cfg={
                    'k_list':[1.4],
                    'fs_feat':'feat',
                    # 'lp_type':'freq_eca',
                    'lp_type':'freq_channel_att',
                    # 'lp_type':'freq_channel_se',
                    # 'lp_type':'freq',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'channel_res':False,
                    'spatial_group':1,
                    'compress_ratio':16,
                    # 'lowfreq_att':False,
                },
                # att_pos='after_bn3',
                # att_pos='after_relu1',
                att_pos='after_conv2',
                train_first_stage=False, 
                train_last_stage=True, 
                **kwargs):
        super().__init__(att=att, 
                         att_cfg=att_cfg, 
                         att_pos=att_pos,
                         train_first_stage=train_first_stage, 
                         train_last_stage=train_last_stage, 
                         **kwargs)
        
from functools import partial

@BACKBONES.register_module()
class NyResNet(ResNetV1cWithBlur):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (NyBottleneck, (3, 4, 6, 3)),
        101: (NyBottleneck, (3, 4, 23, 3)),
        152: (NyBottleneck, (3, 8, 36, 3))
    }
    def __init__(self, att_pos='after_conv2', att_cfg=None, *args, **kwargs):
        # if att_cfg is not None:
            # self.arch_settings[self.depth][0] = partial(self.arch_settings[self.depth][0], att_cfg=att_cfg, att_pos=att_pos)
        super().__init__(*args, **kwargs)

@BACKBONES.register_module()
class NyResNetFreezePretrain(NyResNet):
    def __init__(self, not_freeze_list=['att'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.not_freeze_list = not_freeze_list
        self._custom_freeze_stages()

    def _custom_freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            # for param in m.parameters():
                # param.requires_grad = False
            for name, param in m.named_parameters():
                if any([i in name for i in self.not_freeze_list]):
                    m.train()
                    param.requires_grad = True
                    pass
                else:
                    m.eval()
                    param.requires_grad = False
                print(name, param.requires_grad)

@BACKBONES.register_module()
class ResNetFreqMix(ResNetV1cWithBlur):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fm1 = FrequencySelection(64*self.block.expansion, k_list=[1.4], fs_feat='feat', lp_type='freq_channel_se', act='sigmoid', channel_res=False,)
        self.fm2 = FrequencySelection(128*self.block.expansion, k_list=[1.4], fs_feat='feat', lp_type='freq_channel_se', act='sigmoid', channel_res=False,)
        self.fm3 = FrequencySelection(256*self.block.expansion, k_list=[1.4], fs_feat='feat', lp_type='freq_channel_se', act='sigmoid', channel_res=False,)
        # self.fm4 = FrequencySelection(512*self.block.expansion, k_list=[1.4], fs_feat='feat', lp_type='freq_channel_se', act='sigmoid', channel_res=False,)
        self.fm4 = nn.Identity()

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = self.AdaD[i](x)
            x = res_layer(x)
            fm = getattr(self, f'fm{i+1}')
            x = fm(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
@BACKBONES.register_module()
class ResNetV1cSaliencySampler(ResNetV1c):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    def __init__(self, log_dir=None,  
        sampler_num=3,
        sampler_strides=(1, 1, 1), 
        sampler_paddings=(5, 5, 5), 
        sampler_fwhm=(3, 3, 3),  
        sampler_mode=('avgpsp_semantic_edge', 'avgpsp_semantic_edge', 'avgpsp_semantic_edge'),  
        align_corners=True,
        psp_ratio=4,
        use_avg=(True, True, True), 
        use_checkpoint=True,
        use_lpf=False,
        lpf_kernel=3,
        lpf_group=1,
        lpf_dilation=1,
        lpf_mode = 'feat',
        **kwargs):
        # super().__init__(deep_stem=True, avg_down=True, **kwargs)
        super().__init__(**kwargs)
        if kwargs['depth'] < 50:
            sampler_channels = [64, 128, 256]
        else:
            sampler_channels = [256, 512, 1024]
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.sampler_num = sampler_num
        self.use_checkpoint = use_checkpoint
        self.use_lpf = use_lpf
        self.lpf_group = lpf_group
        self.lpf_kernel = lpf_kernel
        self.lpf_dilation = lpf_dilation
        self.SS = nn.ModuleList([
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SaliencySampler(feat_dim=sampler_channels[0], stride=sampler_strides[0], padding_size=sampler_paddings[0], fwhm=sampler_fwhm[0], sampler_mode=sampler_mode[0], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=3 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[1], stride=sampler_strides[1], padding_size=sampler_paddings[1], fwhm=sampler_fwhm[1], sampler_mode=sampler_mode[1], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=2 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[2], stride=sampler_strides[2], padding_size=sampler_paddings[2], fwhm=sampler_fwhm[2], sampler_mode=sampler_mode[2], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=1 else nn.Identity(),
        ])
        if use_lpf:
            self.lpf_mode = lpf_mode
            if lpf_mode == 'flc':
                self.LPF = nn.ModuleList([
                    nn.Identity(),
                    FLC_Pooling(),
                    FLC_Pooling(),
                    FLC_Pooling(),
                ])
            else:
                if ('feat' in lpf_mode) and ('offset' in lpf_mode):
                    LPF_channels = [i + 4 for i in sampler_channels]
                elif ('feat' in lpf_mode):
                    LPF_channels = [i for i in sampler_channels]
                elif ('offset' in lpf_mode):
                    LPF_channels = [4 for i in sampler_channels]
                else:
                    raise NotImplementedError
                self.LPF = nn.ModuleList([
                    nn.Identity() ,
                    LocalPixelRelationConv(LPF_channels[0], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                    LocalPixelRelationConv(LPF_channels[1], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                    LocalPixelRelationConv(LPF_channels[2], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                    # Downsample_PASA_group_softmax(LPF_channels[0], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                    # Downsample_PASA_group_softmax(LPF_channels[1], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                    # Downsample_PASA_group_softmax(LPF_channels[2], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                ])

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            # x = self.stem(x)
            if self.use_checkpoint: x.requires_grad = True
            x = checkpoint(self.stem, x) if self.use_checkpoint else self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        reversed_outs = []
        grids = []
        smaps = []
        x_sampleds = []
        # maxpool -> stage1 (4x) -> SS1, stage2 (8x) -> SS2, stage3 (16x) -> SS3, stage4 (32x) -> seghead
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 0:
                x = self.SS[i](x) # maxpooling
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x)
                else:
                    x = res_layer(x) 
                grid = None
                saliency = None
                x_sampled = None
                _x_sampled = None
            elif i in (1, 2, 3)[-self.sampler_num:]:
                # _, _, h, w = x.shape
                if self.training:
                    _, _x_sampled, _, _ = self.SS[i](x.detach(), sample_with_grid=True)
                x, x_sampled, saliency, grid = self.SS[i](x, sample_with_grid=True)
                if self.use_lpf:
                    if self.lpf_mode == 'flc':
                        x_sampled = self.LPF[i](x)
                    else:
                        offset = self.SS[i].get_offset(grid)
                        if ('feat' in self.lpf_mode) and ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](torch.cat([x_sampled, offset.detach()], dim=1), x_sampled, use_checkpoint=True)
                        elif ('feat' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](x_sampled, x_sampled, use_checkpoint=True)
                        elif ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](offset, x_sampled, use_checkpoint=True)
                        else:
                            raise NotImplementedError
                # x_sampled = self.SS[i].sample_with_grid(x, grid)
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x_sampled)
                else:
                    x = res_layer(x_sampled)  
            else:
                x = res_layer(x)  
                grid = None
                saliency = None
                # raise NotImplementedError         

            if i in self.out_indices:
                # print(i)
                grids.append(grid)
                smaps.append(saliency)
                outs.append(x)
                x_sampleds.append(_x_sampled)
                # reversed_outs.append(x)

        # return tuple(outs)
        # return tuple(outs), tuple(grids), tuple(smaps)
        return outs, grids, {'smaps':smaps, 'x_sampleds':x_sampleds}
    

