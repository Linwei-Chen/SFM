# -*- coding: utf-8 -*-
"""SFM 接入 mmdetection / mmsegmentation 标准 FPN 的示例。

==============================================================================
一、SFM 在 FPN 网络中的两个接入点
==============================================================================

SFM（Spatial Frequency Modulation, TPAMI 2025）作用于特征金字塔的两个位置：

  接入点 A【下采样 / backbone】：ARS 自适应重采样（SaliencySampler）
      替换 backbone 相邻 stage 之间的普通下采样（maxpool / stride-2 卷积）。
      下采样前先把高频调制到低频，再用非均匀网格采样——重要区域采样密、
      平坦区域采样疏。完整参考实现见
      ``SFM_mask2former/mmseg_custom/models/backbones/resnet_as.py``
      （ResNetV1cSaliencySampler）与 ``swin_as.py``（Swin 版）。

  接入点 B【上采样 / FPN 层间】：SFM 对齐上采样（本文件 SFMFPN）
      替换 FPN top-down 路径中的双线性/最近邻上采样相加：
          标准 FPN:   laterals[i-1] = laterals[i-1] + resize(laterals[i])
          SFM 替换为: laterals[i-1] = laterals[i-1] + LPRMAlignUpModule(
                          x_low=laterals[i], guidance_high_aligned=laterals[i-1])
      即用 LPRM 邻域权重聚合（k² 邻域 unfold + softmax，CARAFE 式内容感知重组）
      完成 2× 放大，并参考同分辨率的高层特征（guidance）决定对齐方向。

==============================================================================
二、如何接入（三步）
==============================================================================

以 mmdetection 2.x（mmcv-full 1.x）为例，mmsegmentation 0.x 完全相同；mmdet 3.x
/mmseg 1.x 只需改第 1 步的 import（见下）。

第 1 步：把本文件复制到框架的 necks 目录，例如
    mmdetection/mmdet/models/necks/SFM_FPN.py
并按框架版本修改顶部 import（本文件已做 try-import 自适应）：
    mmdet 2.x / mmseg 0.x :  from mmcv.runner import BaseModule; from mmcv.cnn import ConvModule
    mmdet 3.x / mmseg 1.x :  from mmengine.model import BaseModule; from mmcv.cnn import ConvModule

第 2 步：注册（在 necks/__init__.py 或本文件末尾）
    from .SFM_FPN import SFMFPN
    NECKS.register_module()(SFMFPN)     # mmdet 2.x / mmseg 0.x
    # mmdet 3.x / mmseg 1.x 直接在类上加 @MODELS.register_module()

第 3 步：改配置。以 Faster R-CNN R-50-FPN 为例（其余 FPN 配置同理，只改 neck）：
    neck=dict(
        type='SFMFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        sfm_compress_ratio=16)   # SFM 权重预测前的通道压缩比

已验证（2026-10-15）：继承 pip 安装的官方 ``mmdet.models.necks.FPN``（2.28.2）
的子类在 GPU 上 forward/backward 通过（4 级金字塔 256/512/1024/2048 → 4×256），
验证脚本见 docs/SFM开源整理/2026-10-15-核心算子定型与即插即用包设计.md。

==============================================================================
三、不使用框架时的纯 PyTorch 用法
==============================================================================

见本文件 ``__main__`` 演示：拿 4 级金字塔特征，逐级调用 LPRMAlignUpModule
完成 top-down 融合，10 行以内。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 框架自适应 import（复制到框架内时按上面第 1 步核对即可）——
try:  # mmdet 2.x / mmseg 0.x
    from mmcv.runner import auto_fp16
except Exception:
    # mmdet 3.x / mmseg 1.x / 纯 PyTorch：fp16 钩子改由 fp16_cfg 处理，
    # 占位装饰器需同时支持 @auto_fp16 与 @auto_fp16() 两种用法
    def auto_fp16(f=None):
        def deco(fn):
            return fn
        return deco if f is None else f

try:  # 官方 FPN 基类（放进框架时用）
    from mmseg.models.necks.fpn import FPN as _OfficialFPN  # mmseg 0.x
except Exception:
    try:
        from mmdet.models.necks.fpn import FPN as _OfficialFPN  # mmdet 2.x/3.x
    except Exception:
        _OfficialFPN = None

from SFM_Align import LPRMAlignUpModule, AlignModule, compute_real_grids


def _register_necks(cls):
    """尽力注册到当前框架的 NECKS/MODELS 注册表（失败则静默跳过）。"""
    try:
        from mmcv.cnn import NECKS  # mmdet 2.x / mmseg 0.x
        NECKS.register_module()(cls)
    except Exception:
        try:
            from mmseg.registry import MODELS  # mmseg 1.x
            MODELS.register_module()(cls)
        except Exception:
            try:
                from mmdet.registry import MODELS  # mmdet 3.x
                MODELS.register_module()(cls)
            except Exception:
                pass
    return cls


@_register_necks
class SFMFPN(torch.nn.Module):
    """带 SFM 对齐上采样的 FPN（子类化官方 FPN，仅替换 top-down 融合）。

    本文件为便于单文件演示直接继承 ``torch.nn.Module`` 并复制了官方 FPN 的
    前向逻辑；放进框架时把基类改为官方 ``FPN``（mmdet: mmdet.models.necks.FPN，
    mmseg: mmseg.models.necks.FPN）并删除复制的 __init__ 通用部分、保留
    sfm 相关部分即可（差异处均有 `# [SFM]` 注释标记）。

    Args:
        in_channels (list[int]): backbone 各级输出通道。
        out_channels (int): FPN 各级统一通道。
        num_outs (int): 输出级数。
        start_level (int): 从第几级开始建 FPN。
        end_level (int): 到第几级为止。
        sfm_compress_ratio (int): SFM 权重预测前的通道压缩比，越大越省算力。
        add_extra_convs / relu_before_extra_convs: 与官方 FPN 含义一致。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 sfm_compress_ratio=16,
                 add_extra_convs=False,
                 relu_before_extra_convs=False):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.add_extra_convs = add_extra_convs
        if end_level == -1:
            backbone_end_level = len(in_channels)
        else:
            backbone_end_level = end_level
        self.backbone_end_level = backbone_end_level

        # —— 官方 FPN 的 lateral / fpn 卷积（保持原样）——
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, backbone_end_level):
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], out_channels, 1))
            self.fpn_convs.append(
                nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                              nn.ReLU(inplace=True)))

        # [SFM] 为每对相邻层级建一个 SFM 对齐上采样模块（替代 resize 相加）
        self.sfm_aligns = nn.ModuleList()
        for _ in range(self.start_level, backbone_end_level - 1):
            self.sfm_aligns.append(
                LPRMAlignUpModule(out_channels, out_channels,
                                  kernel_size=3, compress_ratio=sfm_compress_ratio))

        # 官方 FPN 的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 1) lateral 1×1 投影
        laterals = [lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # 2) top-down 融合（[SFM] 唯一改动点）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 官方 FPN：
            #   laterals[i - 1] = laterals[i - 1] + resize(laterals[i], size=..., mode='nearest')
            # SFM：LPRM 邻域权重聚合 2× 放大，并以同分辨率的高层特征为引导对齐
            laterals[i - 1] = laterals[i - 1] + self.sfm_aligns[i - 1](
                x_low=laterals[i], guidance_high_aligned=laterals[i - 1])

        # 3) 输出（保持官方 FPN 原样）
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                for i in range(used_backbone_levels, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


# ============================================================================
# 接入点 A（下采样）的使用说明与最小示例
# ============================================================================

def sfm_downsample_usage():
    """backbone 下采样处接入 ARS（自适应重采样）的方法说明。

    完整实现：SFM_mask2former/mmseg_custom/models/backbones/interp2d.py
    中的 ``SaliencySampler``（含 sampler_mode='avgpsp_semantic_edge' 等多种模式），
    其在 ResNet / Swin backbone 中的完整接入见同目录 ``resnet_as.py``
    （ResNetV1cSaliencySampler）与 ``swin_as.py``。

    接入方式（以 ResNet 为例，伪代码）::

        from SFM_mask2former.mmseg_custom.models.backbones.interp2d import SaliencySampler

        # 每个 stage 前挂一个 ARS：预测显著图 -> 高斯核加权生成非均匀网格 -> 采样
        self.SS = nn.ModuleList([
            SaliencySampler(feat_dim=256, stride=1, padding_size=5, fwhm=3,
                            sampler_mode='avgpsp_semantic_edge', psp_ratio=4),
            SaliencySampler(feat_dim=512, stride=1, padding_size=5, fwhm=3, ...),
            SaliencySampler(feat_dim=1024, stride=1, padding_size=5, fwhm=3, ...),
        ])

        # forward 中，stage_i 之前：
        x, x_sampled, saliency, grid = self.SS[i](x, sample_with_grid=True)
        x = res_layer(x_sampled)      # stage 自己完成 stride-2 下采样
        # grid/saliency 交给后续模块（网格摊平 AlignModule、L_SHF 语义高频损失）

    说明：
    - stride=1 表示 ARS 本身不改变分辨率，只做"下采样前的调制"
      （把高频能量搬运到低频），随后的 stage 卷积正常降采样；
    - 返回的 ``grid`` 是非均匀采样记录（揉形记录），供 FPN/decoder 端的
      ``AlignModule`` 摊平使用（``compute_real_grids`` 负责逐级复合）；
    - 训练时配合论文的 L_FM 调频损失与 L_SHF 语义高频损失（见论文 Sec.3.4）。
    """
    pass


if __name__ == '__main__':
    # —— 纯 PyTorch 演示：4 级金字塔 + SFM top-down 融合 ——
    torch.manual_seed(0)
    B = 2
    # 模拟 ResNet 金字塔：1/4, 1/8, 1/16, 1/32
    feats = [torch.randn(B, c, s, s)
             for c, s in [(256, 64), (512, 32), (1024, 16), (2048, 8)]]
    in_channels = [f.shape[1] for f in feats]

    neck = SFMFPN(in_channels, out_channels=256, num_outs=4,
                  sfm_compress_ratio=16)
    outs = neck(feats)
    print('输入金字塔:', [tuple(f.shape) for f in feats])
    print('SFMFPN 输出:', [tuple(o.shape) for o in outs])

    # —— 张量级极简用法（不用任何 neck 封装）——
    hr = torch.randn(B, 256, 64, 64)   # 浅层，分辨率高
    lr = torch.randn(B, 256, 32, 32)   # 深层，语义强
    align_up = LPRMAlignUpModule(256, 256, kernel_size=3, compress_ratio=16)
    hr_fused = hr + align_up(x_low=lr, guidance_high_aligned=hr)
    print('张量级融合:', tuple(hr_fused.shape))
