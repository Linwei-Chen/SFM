# -*- coding: utf-8 -*-
"""SFM 上采样/对齐核心算子（纯 PyTorch，零框架依赖）。

包含 FPN / 解码头层间融合所需的三类模块，可 pip 安装后直接调库：

1. ``LPRMAlignUpModule``   —— 邻域权重聚合放大融合（低分辨率 → 高分辨率）
   用 LocalPixelRelation 的思想（k² 邻域权重 + 加权聚合，CARAFE 式内容感知重组）
   把低分辨率特征放大 2× 并与高分辨率引导特征对齐融合。权重由低/高两个压缩
   特征共同预测，上采样时"看着"高层特征决定往哪个方向对齐。这是 FPN top-down
   路径 ``laterals[i-1] + resize(laterals[i])`` 中 resize 的直接替代品（见 ``sfm.fpn``）。

2. ``AlignModule``         —— 网格摊平（非均匀 → 均匀）
   backbone 内的自适应重采样（ARS，见 ``sfm.adaptive_sampling``）会把特征"揉形"：
   重要区域采样密、平坦区域采样疏。本模块利用揉形记录（真实坐标场 real_grid，
   由 ``compute_real_grids`` 逐级复合得到）把特征重新摊回均匀方格。做法：一个小
   卷积预测每个输出像素周围 k=3 个取样点的偏移（offset × sigmoid 门控），取样点
   经 real_grid 翻译成真实位置后构成三角形，用重心坐标公式（几何，无参数）得到
   融合权重，再加权求和。设计动机：教科书式的非均匀反插值需要显式构建/搜索
   三角形，高分辨率下极慢；用可学习偏移直接指定 3 个邻居点即可跳过三角形搜索。

3. ``LocalPixelRelationConv`` —— 标准 LPRM（多空洞率邻域聚合，unfold+softmax）
   低通滤波（dilation=1）与局部关系建模（dilation=2,4,8...）由同一算子实现。

所有实现均为纯 PyTorch，无需 mmcv；与论文实验所用的原始实现数值一致（原始实现
中的 mmcv CARAFE 算子由等价的 ``carafe_reassembly`` 替代，语义已用黑盒探针在
mmcv 1.5.3 上钉死并数值验证到 2.4e-07）。

参考论文:
    Spatial Frequency Modulation for Semantic Segmentation (TPAMI 2025)
    https://arxiv.org/abs/2507.11893
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'carafe_reassembly', 'LocalPixelRelationConv', 'LPRM_efficient',
    'LPRMAlignUpModule', 'AlignModule', 'swap_syncbn_to_bn'
]


# ============================================================================
# 纯 PyTorch 的邻域重组算子（等价替代 mmcv.ops.carafe）
# ============================================================================

def _neighbor_reassembly(value: torch.Tensor, w: torch.Tensor, kernel_size: int,
                         scale_factor: int, dilation: int = 1) -> torch.Tensor:
    """邻域重组的统一实现（含零填充越界语义）。

    对每个输出像素 ``(m, n)``：

        out[m, n] = Σ_{u,v} w[u*k+v, m, n] · value[m//s + (u-p)·d, n//s + (v-p)·d]

    其中 ``p = (k-1)//2``，越界位置补零。``d=1`` 且 ``s`` 任意时与 mmcv.ops.carafe
    的 CUDA/naive 算子（group_size=1）数值一致。

    实现要点：先把 value 按输出分辨率展开（``expanded[m] = value[m//s]``），再做
    零填充与等差切片——偏移 ``(u-p)·d`` 在输入像素尺度，映射到输出像素尺度恰为
    ``(u-p)·d·s``（s 的整数倍，保证整除结构不被破坏）。
    """
    B, C, H, W = value.shape
    k, s, d = kernel_size, scale_factor, dilation
    H_out, W_out = H * s, W * s
    p = (k - 1) // 2

    expanded = value.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)
    padded = F.pad(expanded, [p * d * s] * 4)
    out = torch.zeros(B, C, H_out, W_out, device=value.device, dtype=value.dtype)
    for u in range(k):
        for v in range(k):
            sl = padded[:, :, u * d * s:u * d * s + H_out,
                        v * d * s:v * d * s + W_out]
            out = out + w[:, u * k + v].unsqueeze(1) * sl
    return out


def carafe_reassembly(value: torch.Tensor,
                      mask: torch.Tensor,
                      kernel_size: int = 3,
                      scale_factor: int = 2) -> torch.Tensor:
    """邻域重组（CARAFE 式内容感知上采样/精炼）的纯 PyTorch 实现。

    对每个输出像素 ``(m, n)``，从输入特征的整像素邻域加权重组：

        out[m, n] = Σ_{u,v} mask[u*k+v, m, n] · value[m//s + u - p, n//s + v - p]

    其中 ``p = (k-1)//2``，越界位置补零。该公式与 mmcv.ops.carafe 的 CUDA/naive
    算子在 ``group_size=1`` 时数值一致（已在 mmcv 1.5.3 上用黑盒探针钉死语义并
    数值验证）。

    Args:
        value: 输入特征，shape ``(B, C, H, W)``。
        mask: 逐输出像素的重组权重，shape ``(B, k*k, H*s, W*s)``，
            通道序为行优先（u 为 y 方向、v 为 x 方向）。调用方如未归一化，
            内部会先对通道维做 softmax。
        kernel_size: 邻域大小 k。
        scale_factor: 放大倍数 s（输出分辨率 = 输入 × s；s=1 时为同分辨率精炼）。

    Returns:
        重组后的特征，shape ``(B, C, H*s, W*s)``。
    """
    k, s = kernel_size, scale_factor
    assert mask.shape[1] == k * k, \
        f'mask 通道数应为 k*k={k * k}，实际为 {mask.shape[1]}'
    assert mask.shape[-2] == value.shape[-2] * s and mask.shape[-1] == value.shape[-1] * s, \
        f'mask 空间尺寸应为 ({value.shape[-2] * s}, {value.shape[-1] * s})，' \
        f'实际为 {tuple(mask.shape[-2:])}'
    w = mask.softmax(dim=1)
    return _neighbor_reassembly(value, w, k, s, dilation=1)


class LPRM_efficient(nn.Module):
    """用邻域重组算子实现 LPRM（Local Pixel Relation Module）。

    与原始仓库 ``LocalPixelRelationConv`` 的 ``_forward(x, feat)`` 计算过程一致
    （weight 由外部传入），但重组步骤用 ``carafe_reassembly`` 完成，显存更省。
    原实现依赖 mmcv 的 CARAFE 算子。

    Args:
        channels: value 特征的通道数。
        kernel_size: 邻域大小。
        dilation: 空洞率（>1 时邻域采样点间隔 dilation 个像素）。
        upscale: 放大倍数（1 表示同分辨率精炼）。
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 upscale: int = 1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.upscale = upscale

    def forward(self, mask_pred: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """按外部预测的权重图 ``mask_pred`` 对 ``value_feat`` 做邻域重组。

        Args:
            mask_pred: 重组权重（logits），``(B, k*k, H·upscale, W·upscale)``。
            value_feat: 待重组特征，``(B, C, H, W)``。

        Returns:
            重组后的特征，``(B, C, H·upscale, W·upscale)``。
        """
        w = mask_pred.softmax(dim=1)
        return _neighbor_reassembly(value_feat, w, self.kernel_size,
                                    self.upscale, self.dilation)


# ============================================================================
# 标准 LPRM：多空洞率邻域聚合（unfold + softmax）
# ============================================================================

class Adapad(nn.Module):
    """训练时用 reflect 填充、测试时用 replicate 填充的自适应填充。"""

    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding
        self.reflect_pad = nn.ReflectionPad2d(padding=self.padding)
        self.repeat_pad = nn.ReplicationPad2d(padding=self.padding)

    def forward(self, x):
        if self.training:
            return self.reflect_pad(x)
        return self.repeat_pad(x)


class LocalPixelRelationConv(nn.Module):
    """局部像素关系卷积（LPRM）。

    低通滤波（dilation=1）与局部关系建模（dilation=2,4,8 等）由同一算子实现：
    一个小卷积 ``weight_net`` 为每个输出位置预测 k² 个邻域权重（softmax 归一化），
    再对邻域像素加权聚合。可选自适应残差门控（``use_adaptive_residual``）。

    Args:
        in_channels: 输入通道数。
        kernel_size: 邻域大小 k。
        stride: 输出步长。
        groups: 权重分组数。
        dilation: 空洞率。
        use_adaptive_residual: 是否使用通道门控残差（聚合结果与原特征按
            逐像素 sigmoid 门控加权混合）。
        use_channel: 是否用通道注意力调制邻域权重。
        pad_mode: 填充方式，'reflect' / 'replicate' / 'ada'（训练 reflect、测试 replicate）。
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1,
                 groups: int = 1, dilation: int = 1,
                 use_adaptive_residual: bool = False, use_channel: bool = False,
                 pad_mode: str = 'replicate'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.dilation = dilation
        self.pad = (kernel_size - 1) // 2 * dilation
        self.use_channel = use_channel
        self.use_adaptive_residual = use_adaptive_residual
        self.pad_mode = pad_mode

        if pad_mode == 'reflect':
            self.conv_pad = nn.ReflectionPad2d(padding=self.pad)
        elif pad_mode == 'replicate':
            self.conv_pad = nn.ReplicationPad2d(padding=self.pad)
        elif pad_mode == 'ada':
            self.conv_pad = Adapad(padding=self.pad)
        else:
            raise NotImplementedError(f'未知填充方式: {pad_mode}')

        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=groups * kernel_size ** 2, stride=stride,
                      dilation=dilation, kernel_size=kernel_size, bias=False,
                      padding=0, groups=groups),
            nn.SyncBatchNorm(groups * kernel_size ** 2),
        )

        if use_adaptive_residual:
            self.use_adaptive_residual = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=1,
                          kernel_size=1, bias=False),
                nn.Sigmoid(),
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels // 4, kernel_size=1,
                          bias=False),
                nn.SyncBatchNorm(in_channels // 4),
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4,
                          out_channels=groups * kernel_size ** 2,
                          kernel_size=1, bias=False),
            )

    def _forward(self, x, feat: Optional[torch.Tensor] = None):
        """x 用于预测邻域权重；feat 不为 None 时同时对 feat 做邻域聚合。"""
        _, _, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1

        weight = self.weight_net(self.conv_pad(x))
        if self.use_channel:
            weight = weight * self.channel_net(x)
        pixel_relation = weight

        if feat is not None:
            assert x.size(-1) == feat.size(-1)
            assert x.size(-2) == feat.size(-2)
            b, c, _, _ = feat.shape

            weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow)
            weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

            if self.use_channel:
                tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
                weight = weight * tmp
            weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            weight = weight.reshape(b, self.groups, c // self.groups, oh, ow,
                                    self.kernel_size, self.kernel_size)

            pad_feat = self.conv_pad(feat)
            pad_feat = F.unfold(pad_feat, kernel_size=(self.kernel_size, self.kernel_size),
                                stride=self.stride, dilation=self.dilation)
            pad_feat = pad_feat.reshape(b, self.groups, c // self.groups,
                                        self.kernel_size, self.kernel_size, oh, ow)
            pad_feat = pad_feat.permute(0, 1, 2, 5, 6, 3, 4)
            res = (weight * pad_feat).sum(dim=(-1, -2)).reshape(b, c, oh, ow)

            if self.use_adaptive_residual:
                pixel_weight = self.use_adaptive_residual(x)
                res = pixel_weight * res + (1 - pixel_weight) * feat
            return pixel_relation, res
        return pixel_relation

    def forward(self, x, feat=None, use_checkpoint: bool = False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, feat)
        return self._forward(x, feat)


# ============================================================================
# 核心插值：可学习偏移 + 重心权重的非均匀 → 均匀摊平
# ============================================================================

def _non_uniform_approx_interpolation_with_real_grid(
    feature_to_sample: torch.Tensor,
    offset_predictions: torch.Tensor,
    real_grid_of_feature: torch.Tensor,
    target_h: int,
    target_w: int,
    k: int = 3,
    power: int = 2,
    align_corners: bool = False,
    mode: str = 'barycentric',
    centralized: bool = False,
) -> torch.Tensor:
    """处理累积非均匀性的近似插值（摊平的核心数学）。

    每个目标均匀像素 P 在其附近取 k 个采样点（均匀坐标 = P + offset），
    用 ``real_grid`` 把采样点翻译成真实位置，再按真实位置与 P 的几何关系
    计算融合权重（'barycentric'：k=3 三角形重心坐标；'barycentric_prox'：
    距离倒数幂），最后加权求和 k 次采样结果。

    Args:
        feature_to_sample: 源特征（被揉形的），``(B, C, H, W)``。
        offset_predictions: 预测的像素偏移，``(B, k*2, target_h, target_w)``。
        real_grid_of_feature: 源特征的真实坐标场，``(B, H, W, 2)``。
        target_h, target_w: 目标输出尺寸。
        k: 邻居点数（'barycentric' 模式必须为 3）。
        power: 'barycentric_prox' 模式的距离幂。
        align_corners: 坐标对齐约定。
        mode: 'barycentric' 或 'barycentric_prox'。
        centralized: 是否对偏移做中心化。

    Returns:
        摊平后的特征，``(B, C, target_h, target_w)``。
    """
    B, C, H_src, W_src = feature_to_sample.shape
    device = feature_to_sample.device

    # 1) 目标均匀网格（[-1,1] 归一化坐标）
    if align_corners:
        y_coords = torch.linspace(-1, 1, target_h, device=device)
        x_coords = torch.linspace(-1, 1, target_w, device=device)
    else:
        y_coords = torch.linspace(-1 + 1 / target_h, 1 - 1 / target_h, target_h, device=device)
        x_coords = torch.linspace(-1 + 1 / target_w, 1 - 1 / target_w, target_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
    target_uniform_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # 2) 像素偏移 → 归一化坐标偏移，得到 k 个采样点（均匀坐标系）
    offsets_pixel = offset_predictions.view(B, k, 2, target_h, target_w).permute(0, 3, 4, 1, 2)
    if centralized:
        offsets_pixel = offsets_pixel - offsets_pixel.mean(dim=-2, keepdim=True)
    norm = torch.tensor([target_w / 2, target_h / 2], device=device).view(1, 1, 1, 1, 2)
    offsets = offsets_pixel / norm
    uniform_sample_locations = target_uniform_grid.unsqueeze(-2) + offsets
    uniform_sample_locations = torch.clip(uniform_sample_locations, min=-1, max=1.0)

    # 3) 在源特征上采样 k 个点（注意：采样位置在源特征的"张量坐标"里）
    sampled_features = []
    for i in range(k):
        sampled_feat = F.grid_sample(
            feature_to_sample,
            uniform_sample_locations[:, :, :, i, :],
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        )
        sampled_features.append(sampled_feat)

    # 4) 把采样点翻译到真实（非均匀）坐标系：用采样点坐标去查 real_grid
    final_locations = []
    for i in range(k):
        final_loc = F.grid_sample(
            real_grid_of_feature.permute(0, 3, 1, 2),
            uniform_sample_locations[:, :, :, i, :],
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        ).permute(0, 2, 3, 1)
        final_locations.append(final_loc)
    all_final_locations = torch.stack(final_locations, dim=3)  # (B, H, W, k, 2)

    # 5) 几何权重：三角形重心坐标（无参数）或距离倒数幂
    if mode == 'barycentric_prox':
        real_offsets = all_final_locations - target_uniform_grid.unsqueeze(-2)
        dists = torch.norm(real_offsets, dim=-1, p=2)
        weights = 1.0 / (dists.pow(power) + 1e-8)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        weights = weights.permute(0, 3, 1, 2).unsqueeze(2)  # (B, k, 1, H, W)
    elif mode == 'barycentric':
        assert k == 3, '重心坐标插值模式 (barycentric) 必须使用 k=3 个邻居点。'
        p0 = all_final_locations[..., 0, :]
        p1 = all_final_locations[..., 1, :]
        p2 = all_final_locations[..., 2, :]
        p = target_uniform_grid

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p - p0

        d00 = (v0 * v0).sum(dim=-1)
        d01 = (v0 * v1).sum(dim=-1)
        d11 = (v1 * v1).sum(dim=-1)
        d20 = (v2 * v0).sum(dim=-1)
        d21 = (v2 * v1).sum(dim=-1)
        denom = d00 * d11 - d01 * d01

        w1 = (d11 * d20 - d01 * d21) / (denom + 1e-8)
        w2 = (d00 * d21 - d01 * d20) / (denom + 1e-8)
        w1 = torch.clip(w1, min=0, max=1.0)
        w2 = torch.clip(w2, min=0, max=1.0)
        w0 = 1.0 - w1 - w2
        weights = torch.stack([w0, w1, w2], dim=-1).permute(0, 3, 1, 2).unsqueeze(2)
    else:
        raise ValueError(f'未知的权重模式: {mode}，可选 barycentric / barycentric_prox。')

    # 6) 加权求和
    all_sampled_feats = torch.stack(sampled_features, dim=1)  # (B, k, C, H, W)
    return torch.sum(all_sampled_feats * weights, dim=1)


class AlignModule(nn.Module):
    """非均匀特征摊平模块（可学习偏移 + 重心权重）。

    输入揉形特征与它的真实坐标场，输出站回均匀方格的特征。
    offset 预测器初始化为全零：训练起点近似恒等映射（3 个采样点重合 →
    重心权重退化为 (1,0,0)），可零成本插入任何已有结构。

    Args:
        in_channels: 输入通道数。
        k: 邻居点数（默认 3，'barycentric' 模式）。
        align_corners: 坐标对齐约定。
        compress_ratio: 偏移预测前的通道压缩比（省算力）。
        use_scale: 是否使用 sigmoid 门控缩放偏移。
    """

    def __init__(self, in_channels: int, k: int = 3, align_corners: bool = False,
                 compress_ratio: int = 4, use_scale: bool = True):
        super().__init__()
        self.k = k
        self.align_corners = align_corners

        compressed_channels = in_channels // compress_ratio
        self.compress_conv = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, 1, bias=True),
        )
        self.offset_predictor = nn.Conv2d(compressed_channels, k * 2, 3, padding=1)
        if use_scale:
            self.scale_predictor = nn.Conv2d(compressed_channels, k * 2, 3, padding=1)

        nn.init.constant_(self.offset_predictor.weight, 0)
        nn.init.constant_(self.offset_predictor.bias, 0)

    def forward(self, x: torch.Tensor, real_grid: torch.Tensor) -> torch.Tensor:
        """把揉形特征 ``x`` 按真实坐标场 ``real_grid`` 摊平到均匀方格。"""
        B, C, H, W = x.shape

        x_compressed = self.compress_conv(x)
        if hasattr(self, 'scale_predictor'):
            scale = self.scale_predictor(x_compressed).sigmoid_()
        else:
            scale = 1.0
        offsets = self.offset_predictor(x_compressed) * scale

        x_aligned = _non_uniform_approx_interpolation_with_real_grid(
            x, offsets, real_grid, H, W, self.k,
            align_corners=self.align_corners, mode='barycentric')
        return x_aligned


# ============================================================================
# LPRM 权重聚合放大融合
# ============================================================================

class LPRMAlignUpModule(nn.Module):
    """低分辨率特征 ×2 放大 + 与高分辨率引导特征对齐融合。

    权重由低/高两个压缩特征共同预测（低分辨率预测 ×2 上采样 + 高分辨率预测
    对齐尺寸，相加），再由 ``LPRM_efficient``（邻域重组）完成 2× 放大。
    这是 FPN top-down 融合里 ``resize(低层特征)`` 的即插即用替代品，
    用法见 ``sfm.fpn.SFMFPN``。

    Args:
        low_res_channels: 低分辨率特征通道数。
        high_res_channels: 高分辨率引导特征通道数。
        kernel_size: 邻域大小 k。
        align_corners: 坐标对齐约定。
        compress_ratio: 权重预测前的通道压缩比。
    """

    def __init__(self, low_res_channels: int, high_res_channels: int,
                 kernel_size: int = 3, align_corners: bool = False,
                 compress_ratio: int = 4):
        super().__init__()
        self.align_corners = align_corners
        self.kernel_size = kernel_size

        predictor_in_channels = low_res_channels + high_res_channels
        compressed_channels = predictor_in_channels // compress_ratio

        self.compress_conv_low = nn.Sequential(
            nn.Conv2d(low_res_channels, compressed_channels, 1, bias=True),
        )
        self.compress_conv_high = nn.Sequential(
            nn.Conv2d(high_res_channels, compressed_channels, 1, bias=True),
        )

        # dilation = 1 is enough for FPN structure
        self.lprm = LPRM_efficient(channels=compressed_channels,
                                   kernel_size=kernel_size, upscale=2)
        self.lpr_conv = nn.Conv2d(compressed_channels, kernel_size ** 2, 3, padding=1)

    def forward(self, x_low: torch.Tensor,
                guidance_high_aligned: torch.Tensor) -> torch.Tensor:
        """把 ``x_low`` 放大到 ``guidance_high_aligned`` 的分辨率并返回。

        Args:
            x_low: 低分辨率特征，``(B, C, h, w)``。
            guidance_high_aligned: 高分辨率引导特征，``(B, C, 2h, 2w)``（通常
                已经过网格摊平）。

        Returns:
            放大后的低分辨率特征，``(B, C, target_h, target_w)``。
        """
        B, C, target_h, target_w = guidance_high_aligned.shape
        compressed_low = self.compress_conv_low(x_low)
        compressed_high = self.compress_conv_high(guidance_high_aligned)
        # 邻域权重：低分辨率预测 ×2 上采样 + 高分辨率预测对齐尺寸，相加
        lpr_feat = F.interpolate(self.lpr_conv(compressed_low), scale_factor=2,
                                 mode='bilinear', align_corners=self.align_corners) \
            + F.interpolate(self.lpr_conv(compressed_high),
                            size=(x_low.size(-2) * 2, x_low.size(-1) * 2),
                            mode='bilinear', align_corners=self.align_corners)
        x_low = self.lprm(lpr_feat, x_low)
        x_low_upsampled = F.interpolate(x_low, size=(target_h, target_w),
                                        mode='bilinear', align_corners=self.align_corners)
        return x_low_upsampled


# ============================================================================
# 兼容性助手
# ============================================================================

def swap_syncbn_to_bn(module: nn.Module) -> nn.Module:
    """把模块里的 SyncBatchNorm 就地替换为 BatchNorm2d。

    SyncBN 在非分布式（CPU / 单卡不初始化进程组）环境下前向会直接报错；
    本助手替换后的 BN 与 SyncBN 数学等价（eval 模式）、state_dict 同构，
    因此加载 released checkpoint 前后均可安全调用。GPU 分布式训练时无需调用。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features)
            bn.load_state_dict(child.state_dict())
            setattr(module, name, bn)
        else:
            swap_syncbn_to_bn(child)
    return module
