# -*- coding: utf-8 -*-
"""SFMFPN：可直接在 MMDetection / MMSegmentation 配置里使用的 SFM FPN。

把官方 FPN top-down 路径中的上采样相加：

    标准 FPN:   laterals[i-1] = laterals[i-1] + resize(laterals[i])
    SFM 替换为: laterals[i-1] = laterals[i-1] + LPRMAlignUpModule(
                    x_low=laterals[i], guidance_high_aligned=laterals[i-1])

即用 LPRM 邻域权重聚合（k² 邻域 unfold + softmax，CARAFE 式内容感知重组）
完成 2× 放大，并参考同分辨率的高层特征（guidance）决定对齐方向。其余部分
（lateral 1×1、fpn 3×3、extra levels、fp16 钩子）与官方 FPN 完全一致。

设计：有 mmdet/mmseg 时继承其官方 ``FPN`` 基类（构造逻辑、初始化、ConvModule
支持全部继承，配置零成本迁移）；两者都未安装时回退到内置的纯 PyTorch FPN
（数值行为与官方默认配置一致）。导入本模块即自动注册到已安装框架的
neck 注册表，配置里写 ``type='SFMFPN'`` 即可。

用法（mmseg 0.x/1.x、mmdet 2.x/3.x 通用）::

    # config 里加一行：
    custom_imports = dict(imports=['sfm'], allow_failed_imports=False)
    # 然后把 neck 的 type 换掉：
    neck=dict(type='SFMFPN', in_channels=[...], out_channels=256, num_outs=5)

已验证（2026-10-15，8×GPU 服务器）：继承 pip 安装的官方
``mmdet.models.necks.FPN``（2.28.2）在 GPU 上 forward/backward 通过。

参考论文:
    Spatial Frequency Modulation for Semantic Segmentation (TPAMI 2025)
    https://arxiv.org/abs/2507.11893
"""

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upsampling import LPRMAlignUpModule

__all__ = ['SFMFPN', 'register_sfm', 'FPN_BASE_SOURCE']


# ---------------------------------------------------------------------------
# fp16 钩子（mmdet 2.x 用 mmcv.runner，3.x 用 mmengine.runner，纯 torch 用占位）
# ---------------------------------------------------------------------------
try:
    from mmengine.runner import auto_fp16  # mmdet 3.x / mmseg 1.x
except Exception:
    try:
        from mmcv.runner import auto_fp16  # mmdet 2.x / mmseg 0.x
    except Exception:
        def auto_fp16(f=None):
            def deco(fn):
                return fn
            return deco if f is None else f


# ---------------------------------------------------------------------------
# 官方 FPN 基类：mmdet 优先，其次 mmseg，都无则用内置纯 torch 版
# ---------------------------------------------------------------------------
FPN_BASE_SOURCE = 'none'
try:
    from mmdet.models.necks import FPN as _OfficialFPN  # mmdet 2.x / 3.x
    FPN_BASE_SOURCE = 'mmdet'
except Exception:
    try:
        from mmseg.models.necks import FPN as _OfficialFPN  # mmseg 1.x 独立版
        FPN_BASE_SOURCE = 'mmseg'
    except Exception:
        _OfficialFPN = None

if _OfficialFPN is None:

    class _OfficialFPN(nn.Module):
        """内置纯 PyTorch FPN（仅当 mmdet/mmseg 都不可用时作为基类）。

        复刻 mmdet 2.28.2 官方 FPN 的结构与默认行为（lateral 1×1 无 norm/act、
        fpn 3×3 无 norm/act、Xavier 初始化、extra convs stride-2）。
        conv_cfg/norm_cfg/act_cfg 参数仅为兼容配置签名，非 None 时不支持。
        """

        def __init__(self,
                     in_channels,
                     out_channels,
                     num_outs,
                     start_level=0,
                     end_level=-1,
                     add_extra_convs=False,
                     relu_before_extra_convs=False,
                     no_norm_on_lateral=False,
                     conv_cfg=None,
                     norm_cfg=None,
                     act_cfg=None,
                     upsample_cfg=dict(mode='nearest'),
                     init_cfg=None,
                     **kwargs):
            super().__init__()
            if conv_cfg is not None or norm_cfg is not None or act_cfg is not None:
                raise NotImplementedError('内置 FPN 基类不支持 conv/norm/act 配置，'
                                          '请安装 mmdet 或 mmseg 使用完整功能')
            assert isinstance(in_channels, list)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_ins = len(in_channels)
            self.num_outs = num_outs
            self.relu_before_extra_convs = relu_before_extra_convs
            self.no_norm_on_lateral = no_norm_on_lateral
            self.fp16_enabled = False
            self.upsample_cfg = upsample_cfg.copy()
            self.init_cfg = init_cfg

            if end_level == -1 or end_level == self.num_ins - 1:
                self.backbone_end_level = self.num_ins
                assert num_outs >= self.num_ins - start_level
            else:
                self.backbone_end_level = end_level + 1
                assert end_level < self.num_ins
                assert num_outs == end_level - start_level + 1
            self.start_level = start_level
            self.end_level = end_level
            self.add_extra_convs = add_extra_convs
            assert isinstance(add_extra_convs, (str, bool))
            if isinstance(add_extra_convs, str):
                assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
            elif add_extra_convs:
                self.add_extra_convs = 'on_input'

            self.lateral_convs = nn.ModuleList()
            self.fpn_convs = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level):
                self.lateral_convs.append(nn.Conv2d(in_channels[i], out_channels, 1))
                self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

            extra_levels = num_outs - self.backbone_end_level + self.start_level
            if self.add_extra_convs and extra_levels >= 1:
                for i in range(extra_levels):
                    if i == 0 and self.add_extra_convs == 'on_input':
                        extra_in = self.in_channels[self.backbone_end_level - 1]
                    else:
                        extra_in = out_channels
                    self.fpn_convs.append(
                        nn.Conv2d(extra_in, out_channels, 3, stride=2, padding=1))

            # 与官方默认 init_cfg（Xavier uniform）一致
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# SFMFPN
# ---------------------------------------------------------------------------

class SFMFPN(_OfficialFPN):
    """带 SFM 对齐上采样的 FPN（继承官方 FPN，仅替换 top-down 融合）。

    Args:
        in_channels (list[int]): backbone 各级输出通道。
        out_channels (int): FPN 各级统一通道。
        num_outs (int): 输出级数。
        start_level / end_level / add_extra_convs / relu_before_extra_convs /
            no_norm_on_lateral / conv_cfg / norm_cfg / act_cfg / upsample_cfg /
            init_cfg: 与官方 FPN 含义完全一致，直接透传。
        sfm_compress_ratio (int): SFM 权重预测前的通道压缩比，越大越省算力
            （released Mask2Former 配置用 16）。
        sfm_kernel_size (int): LPRM 邻域大小 k（默认 3）。
        sfm_align_corners (bool): SFM 内部插值的坐标对齐约定。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=None,
                 sfm_compress_ratio=16,
                 sfm_kernel_size=3,
                 sfm_align_corners=False,
                 **kwargs):
        # init_cfg 为 None 时不透传，保留各框架官方 FPN 的默认初始化
        _extra = dict(kwargs)
        if init_cfg is not None:
            _extra['init_cfg'] = init_cfg
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            **_extra)
        self.fp16_enabled = False  # mmdet 2.x 需要（3.x 由 wrap_fp16_model 控制）
        self.sfm_compress_ratio = sfm_compress_ratio
        self.sfm_kernel_size = sfm_kernel_size
        self.sfm_align_corners = sfm_align_corners

        # [SFM] 为每对相邻层级建一个对齐上采样模块（替代 top-down 的 resize）。
        # 融合只发生在 (i-1, i) 相邻对上，共 N-1 对；最高层不接收上采样，
        # 不建模块（避免 DDP 的 unused parameters 报错）。
        self.sfm_aligns = nn.ModuleList([
            LPRMAlignUpModule(out_channels, out_channels,
                              kernel_size=sfm_kernel_size,
                              align_corners=sfm_align_corners,
                              compress_ratio=sfm_compress_ratio)
            for _ in range(max(len(self.lateral_convs) - 1, 0))
        ])

    @auto_fp16()
    def forward(self, inputs):
        """与官方 FPN 前向逐行对应，唯一差异在 top-down 融合（[SFM] 标记处）。"""
        assert len(inputs) == len(self.in_channels)

        # 1) lateral 1×1 投影（官方原样）
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2) top-down 融合 —— [SFM] 唯一改动点
        #    官方:  laterals[i-1] = laterals[i-1] + resize(laterals[i])
        #    SFM:   LPRM 邻域权重聚合 2× 放大，以同分辨率的高层特征为引导对齐
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self.sfm_aligns[i - 1](
                x_low=laterals[i], guidance_high_aligned=laterals[i - 1])

        # 3) 输出（官方原样：fpn 3×3 + extra levels）
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


# ---------------------------------------------------------------------------
# 注册：导入本模块即自动注册到已安装框架的 neck 注册表
# ---------------------------------------------------------------------------

# (标签, 模块路径, 注册表属性名)。同一框架的别名注册表重复注册会抛 KeyError，已捕获。
_REGISTRATION_TARGETS = [
    ('mmdet.NECKS (mmdet 2.x)', 'mmdet.models.builder', 'NECKS'),
    ('mmdet.MODELS (mmdet 3.x)', 'mmdet.registry', 'MODELS'),
    ('mmseg.NECKS (mmseg 0.x)', 'mmseg.models.builder', 'NECKS'),
    ('mmseg.MODELS (mmseg 1.x)', 'mmseg.registry', 'MODELS'),
]


def register_sfm(cls=SFMFPN):
    """把 SFM neck 注册到当前已安装框架的注册表（尽力而为，幂等）。

    Returns:
        成功注册到的目标列表，如 ``['mmdet.NECKS (mmdet 2.x)']``；
        未安装任何框架时返回空列表（纯 torch 用法不受影响）。
    """
    done = []
    for label, mod_path, attr in _REGISTRATION_TARGETS:
        try:
            registry = getattr(importlib.import_module(mod_path), attr)
            registry.register_module()(cls)
            done.append(label)
        except KeyError:
            pass  # 已注册过（幂等）
        except Exception:
            pass  # 框架未安装或版本不支持
    return done


_SFM_REGISTERED_TO = register_sfm()
