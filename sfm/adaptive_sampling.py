# -*- coding: utf-8 -*-
"""最小化自适应重采样 ARS（Adaptive Resampling，论文中的 SaliencySampler）。

这是论文 SFM 在 backbone 下采样处（接入点 A）的核心算子的精简可移植版：
下采样前先把特征高频调制到低频（等式 2/3 的频域搬移），再用非均匀网格采样——
重要区域采样密、平坦区域采样疏。流程四步：

  1. ``saliency_conv``：一个小卷积从特征预测 1 通道显著图（saliency）；
  2. 对显著图逐像素 softmax × H·W，得到归一化密度（总质量 = 像素数）；
  3. ``create_grid``：用固定高斯核（fwhm 控制宽度）对密度做加权质心，
     得到每个输出位置的非均匀采样坐标（[-1,1] 归一化，图像边界钉回均匀）；
  4. ``F.grid_sample`` 按网格重采样特征（padding_mode='border'）。

AR 本身不改变分辨率（stride=1 时只做"下采样前的调制"），随后的 stage 卷积
正常降采样；返回的 ``grid`` 是揉形记录，供上游 ``AlignModule`` 摊平时用
``compute_real_grids`` 逐级复合。

与原始仓库 ``interp2d.SaliencySampler`` 的关系：
  - 支持的 sampler_mode 输出与原版**逐键、逐值一致**（state_dict 可直接互载，
    服务器上已做数值对齐验证）；
  - 逆插值 ``reverse``/``reverse_with_inv_grid``/``reverse_grid`` 已按原版
    逐行移植（qhull 非均匀上采样，FCN style 模型 MSAU 推理路径，见
    :mod:`sfm.interpolate`）；
  - 删去了原版仅用于调试的部分：TensorBoard 可视化（log_dir）、以及
    aliasing/softpool/cbam/lap/cossim 等未在 released 配置中使用的模式。
    需要完整版请用仓库 ``SFM_mask2former/mmseg_custom``。

参考论文:
    Spatial Frequency Modulation for Semantic Segmentation (TPAMI 2025)
    https://arxiv.org/abs/2507.11893
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import List, Optional, Tuple

__all__ = ['SaliencySampler', 'ARS', 'makeGaussian', 'PSP', 'PSP_Module',
           'LHPFConv3', 'compute_real_grids']


def makeGaussian(size, fwhm=3, center=None):
    """生成边长 ``size`` 的二维高斯核（float32 ndarray）。

    ``fwhm`` 是半高全宽，可理解为高斯斑的有效半径：越小核越尖，
    非均匀网格向显著位置的偏移越局部化。
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class PSP_Module(nn.Module):
    """PSP 金字塔池化单元：AdaptiveAvgPool(bin) → 1×1 conv → BN → ReLU → 放大回原尺寸。"""

    def __init__(self, in_channels, out_channels, bin):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(bin)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()
        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size[2:], mode='bilinear', align_corners=True)
        return x


class PSP(nn.Module):
    """4 路金字塔池化拼回输入（全局上下文注入，用于显著图预测）。"""

    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.psp1 = PSP_Module(in_channels, out_channels, bins[0])
        self.psp2 = PSP_Module(in_channels, out_channels, bins[1])
        self.psp3 = PSP_Module(in_channels, out_channels, bins[2])
        self.psp4 = PSP_Module(in_channels, out_channels, bins[3])

    def forward(self, x):
        x1 = self.psp1(x)
        x2 = self.psp2(x)
        x3 = self.psp3(x)
        x4 = self.psp4(x)
        out = torch.cat([x, x1, x2, x3, x4], dim=1)
        return out


class LHPFConv3(nn.Module):
    """可学习高通滤波器（Learnable High Pass Filter，3×3 深度卷积实现）。

    卷积核经 softmax 归一化成一个可学习的低通核（平滑），再从单位核
    （residual=True 时为 2×单位核）中减去它得到高通核：
        高通信号 = 原始信号 - 低通信号（residual=True 时保留 2× 残差主通路）。
    高频能量正是 SFM 调制的对象，因此显著图预测前先做一次高通提取。
    """

    def __init__(self, channels: int = 3, stride: int = 1, padding: int = 1,
                 residual: bool = False):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding

        # 低通核基底（全 1 环形 + 0 中心），softmax 后成为可学习低通核
        lpf_kernel_base = torch.tensor([
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]
        ], dtype=torch.float32)
        lpf_kernel_base = lpf_kernel_base.unsqueeze(0).unsqueeze(0)
        lpf_kernel_base = lpf_kernel_base.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=lpf_kernel_base, requires_grad=True)

        identity_kernel = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ], dtype=torch.float32)
        if residual:
            identity_kernel = identity_kernel * 2
        self.register_buffer('identity_kernel', identity_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        weight_softmax = self.weight.reshape(self.channels, 1, -1).softmax(dim=-1)
        low_pass_kernel = weight_softmax.reshape(self.channels, 1, 3, 3)
        high_pass_kernel = self.identity_kernel - low_pass_kernel
        output = F.conv2d(x, high_pass_kernel,
                          padding=self.padding, groups=self.channels, stride=self.stride)
        return output


class SaliencySampler(nn.Module):
    """自适应重采样 ARS（= 原始仓库的 SaliencySampler，精简移植版）。

    用法（在 backbone 的 stage 前插入，stride=1 表示 ARS 自身不降采样）::

        ss = SaliencySampler(feat_dim=256, stride=1, padding_size=5, fwhm=3,
                             sampler_mode='avg_lhpf_psp', psp_ratio=16,
                             align_corners=False)
        x, x_sampled, saliency, grid = ss(x, sample_with_grid=True)
        x = stage_conv(x_sampled)   # stage 自己完成 stride-2 下采样
        # grid/saliency 交给后续模块：网格摊平 AlignModule、L_SHF 语义高频损失

    Args:
        feat_dim: 输入特征通道数。
        stride: ARS 内部池化步长（backbone 接入用 1：只调制不降采样；
            独立使用时可用 2 直接降分辨率）。
        padding_size: 高斯加权质心的搜索半径（像素）。越大网格可偏移越远。
        fwhm: 高斯核半高全宽（像素）。越小，采样越"盯着"最显著的位置。
        dilation: 显著图卷积的空洞率。
        align_corners: 坐标对齐约定（与 F.grid_sample 一致；released 配置为 False）。
        sampler_mode: 显著图预测方式，支持
            - ``'avg_lhpf_psp'``（released Mask2Former 配置）：AvgPool → 可学习高通 → PSP → 1×1 conv
            - ``'avgpsp_semantic_edge'`` 等 semantic_edge 族：AvgPool → PSP → conv
            - ``'conv' / 'conv3' / 'avg_conv' / 'avg_conv3'``：池化 + 单卷积（最简）
            - ``'none'``：不预测显著图，恒等采样（均匀网格）
        padding_mode: 网格计算时显著图的填充方式（'reflect' 优于 'replicate'，
            避免边界采样偏置）。
        psp_ratio: PSP 通道压缩比。
        use_checkpoint: 是否对显著图卷积用梯度检查点（省显存）。
    """

    def __init__(self, feat_dim=256, stride=2, padding_size=31, fwhm=32,
                 dilation=1, use_checkpoint=False, detach=False,
                 align_corners=True, sampler_mode='avgpsp_semantic_edge',
                 padding_mode='reflect', psp_ratio=4, log_dir=None,
                 psp_bins=None):
        super().__init__()
        # log_dir 仅为原始 SaliencySampler 的配置兼容参数（原用于 TensorBoard
        # 可视化，最小版已移除），接受并忽略，保证配置 dict 可零成本替换
        self.feat_dim = feat_dim
        self.stride = stride
        self.detach = detach
        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.fwhm = fwhm
        self.eps = 1e-16
        if psp_bins is None:
            # 默认与 Mask2Former released 版一致：bins 随 align_corners 适配
            psp_bins = [1, 2, 3, 7] if align_corners else [1, 2, 3, 6]
        else:
            psp_bins = list(psp_bins)
        # 注意：UPerNet/InternImage 训练树未传 bins（恒用默认 [1,2,3,7]），
        # 复现该线数值时请显式传 psp_bins=[1,2,3,7]（state_dict 不受影响）
        self.psp_bins = psp_bins
        # 逆插值（qhull）按 h/w 懒建并缓存，见 reverse()
        self.reverse_sampler = None

        if 'softpool' in sampler_mode:
            raise NotImplementedError('minimal 版不支持 softpool，请用完整仓库 SFM_mask2former')
        if 'semantic_edge' in sampler_mode:
            use_psp = ('psp' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            if use_avg:
                POOL = nn.AvgPool2d(kernel_size=2 * stride + 1, stride=stride, padding=stride)
            else:
                POOL = nn.Identity()
            _in_channels = (feat_dim + feat_dim // psp_ratio * 4) if use_psp else feat_dim
            self.saliency_conv = nn.Sequential(
                POOL,
                PSP(feat_dim, feat_dim // psp_ratio, bins=psp_bins) if use_psp else nn.Identity(),
                nn.Conv2d(_in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1)
                if use_conv3 else nn.Conv2d(_in_channels, 1, kernel_size=1, padding=0, stride=1),
            )
        elif 'lhpf' in sampler_mode:
            use_psp = ('psp' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            _in_channels = (feat_dim + feat_dim // psp_ratio * 4) if use_psp else feat_dim
            if use_avg:
                POOL = nn.AvgPool2d(kernel_size=2 * stride + 1, stride=stride, padding=stride)
            else:
                POOL = nn.Identity()
            self.saliency_conv = nn.Sequential(
                POOL,
                LHPFConv3(channels=feat_dim, stride=1, padding=1, residual=True),
                PSP(feat_dim, feat_dim // psp_ratio, bins=psp_bins) if use_psp else nn.Identity(),
                nn.Conv2d(_in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1)
                if use_conv3 else nn.Conv2d(_in_channels, 1, kernel_size=1, padding=0, stride=1),
            )
        elif 'conv' in sampler_mode:
            use_avg = ('avg' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            self.saliency_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size=2 * stride + 1, stride=stride, padding=stride)
                if use_avg else nn.Identity(),
                nn.Conv2d(feat_dim, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1)
                if use_conv3 else nn.Conv2d(feat_dim, 1, kernel_size=1, padding=0, stride=1),
            )
        elif 'none' in sampler_mode:
            self.saliency_conv = None
        else:
            raise NotImplementedError(
                f'minimal 版 sampler_mode 支持 semantic_edge/lhpf/conv/none 族，'
                f'不支持: {sampler_mode}（cbam/lap/cossim 等请用完整仓库 SFM_mask2former）')

        self.padding_size = padding_size

        # 固定高斯核（不可学习），fwhm 控制质心计算的局部性
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=fwhm))
        self.gaussian_weights = gaussian_weights
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights
        self.filter.weight.requires_grad = False

        # 不同输入尺寸的坐标网格缓存（CPU 张量，用时 .to(device)）
        self.coord_grid = {}

    # ------------------------------------------------------------------
    # 坐标网格
    # ------------------------------------------------------------------

    def gen_coord_grid(self, h, w, device=None):
        """带 padding 的均匀坐标网格（[-p, w+p) → 归一化），shape ``[2, h+2p, w+2p]``。"""
        coord_grid = self.coord_grid.get(f'{h}x{w}', None)
        if coord_grid is None:
            if self.align_corners:
                x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / (w - 1.0)
                y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / (h - 1.0)
            else:
                x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / w + 0.5 / w
                y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / h + 0.5 / h
            coord_grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0)
            self.coord_grid[f'{h}x{w}'] = coord_grid
        return coord_grid.to(device).detach()

    def gen_coord_grid_without_pad(self, h, w, device=None):
        """去掉 padding 边缘的均匀坐标网格，shape ``[2, h, w]``。"""
        coord_grid_without_pad = self.gen_coord_grid(h, w, device)
        return coord_grid_without_pad[:, self.padding_size:-self.padding_size,
                                      self.padding_size:-self.padding_size]

    def get_offset(self, grid, cell_encode=True):
        """从采样网格中提取相对均匀网格的绝对偏移（调试/LPF 用）。

        Returns:
            ``cell_encode=True`` 时 shape ``(B, 4, H, W)``：[offset_x, offset_y, cell_x, cell_y]；
            否则 ``(B, 2, H, W)``。
        """
        b, _, h, w = grid.shape
        P = self.gen_coord_grid_without_pad(h, w, device=grid.device) * 2 - 1.0
        P = P[None, :, :, :].expand(b, 2, h, w).to(grid.device)
        offset = grid - P
        if cell_encode:
            cell = torch.ones_like(offset).to(offset.device)
            cell[:, 0] = 2 / w
            cell[:, 1] = 2 / h
            return torch.cat([offset, cell], dim=1)
        return offset

    # ------------------------------------------------------------------
    # 核心：显著图 → 非均匀网格（高斯加权质心）
    # ------------------------------------------------------------------

    def create_grid(self, x, h, w):
        """由显著图计算非均匀采样网格。

        x: 显著图密度 ``(B, 1, h, w)``（逐像素 softmax × H·W 后的）。
        返回: grid ``(B, 2, h, w)``，[x, y] 顺序，[-1,1] 归一化，
        图像边界一行/列钉回均匀坐标（防止边界漂移）。
        """
        # reflect 填充显著图，避免等式 2/3 的质心计算偏向图像中心
        x = F.pad(x, pad=[self.padding_size] * 4, mode=self.padding_mode)

        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size
        # P：每个 padded 位置的均匀坐标（[-1,1]）
        P = self.gen_coord_grid(h, w, device=x.device)[None, ]
        P = P.expand(x.size(0), 2, padded_h, padded_w).to(x.device)

        x_cat = torch.cat((x, x), 1)  # [B, 2, padded_h, padded_w]
        # 论文等式 2/3 的分母：密度的局部和
        p_filter = self.filter(x).float()
        # 分子：坐标 × 密度 的局部和
        x_mul = torch.mul(P, x_cat).view(-1, 1, padded_h, padded_w)
        all_filter = self.filter(x_mul).view(-1, 2, h, w)

        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, h, w)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, h, w)

        # 加权质心 ∈ [0,1] → [-1,1]
        x_filter = x_filter / (p_filter + self.eps)
        y_filter = y_filter / (p_filter + self.eps)
        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        clamp_value_x = 1 if self.align_corners else (1 - 0.5 / w * 2)
        clamp_value_y = 1 if self.align_corners else (1 - 0.5 / h * 2)
        xgrids = torch.clamp(xgrids, min=-clamp_value_x, max=clamp_value_x)
        ygrids = torch.clamp(ygrids, min=-clamp_value_y, max=clamp_value_y)

        xgrids = xgrids.view(-1, 1, h, w)
        ygrids = ygrids.view(-1, 1, h, w)
        grid = torch.cat((xgrids, ygrids), 1)

        # 边界钉回均匀坐标：边界像素的采样点不允许越出均匀位置
        grid[:, 1, 0, :] = P[:, 1, self.padding_size, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, 1, -1, :] = P[:, 1, -self.padding_size - 1, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, 0, :, 0] = P[:, 0, self.padding_size:-self.padding_size, self.padding_size] * 2 - 1.0
        grid[:, 0, :, -1] = P[:, 0, self.padding_size:-self.padding_size, -self.padding_size - 1] * 2 - 1.0
        return grid  # in x, y

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(self, feature, sample_with_grid=True):
        """返回 ``(feature, feature_sampled, saliency, grid)``。

        ``feature`` 原样返回（ARS 不在原地修改特征）；``feature_sampled`` 是
        按非均匀网格重采样后的特征（``sample_with_grid=False`` 时为 None）；
        ``saliency`` 是归一化密度图（``sampler_mode='none'`` 时为 None）。
        """
        if self.saliency_conv is None:
            b, _, h, w = feature.shape
            feature_sampled = feature
            saliency = None
            grid = self.gen_coord_grid_without_pad(h, w, device=feature.device)
            grid = grid[None, ].repeat(b, 1, 1, 1) * 2 - 1
            return feature, feature_sampled, saliency, grid

        if self.use_checkpoint:
            saliency = checkpoint(self.saliency_conv, feature)
        else:
            saliency = self.saliency_conv(feature)
        _, c, h, w = saliency.shape
        # 逐像素 softmax 归一化成密度（总质量 = 像素数）
        saliency = saliency.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w) * h * w
        grid = self.create_grid(saliency, h, w)

        if sample_with_grid:
            if self.detach:
                feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1).detach(),
                                                mode='bilinear', align_corners=self.align_corners,
                                                padding_mode="border")
            else:
                feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1),
                                                mode='bilinear', align_corners=self.align_corners,
                                                padding_mode="border")
        else:
            feature_sampled = None
        return feature, feature_sampled, saliency, grid

    # ------------------------------------------------------------------
    # 逆插值（qhull 非均匀上采样）：FCN style 模型 MSAU 推理路径。
    # 逐行移植自原始 SaliencySampler.reverse（仅去 TensorBoard），实现
    # 在 sfm.interpolate 中；qhull 后端自动探测，详见该模块 docstring。
    # ------------------------------------------------------------------

    def reverse(self, sampled_feat, grid, h, w):
        """把非均匀采样特征重建回 (h, w) 均匀网格（qhull 三点重心插值）。"""
        from .interpolate import reverse_feat

        reversed_feats, self.reverse_sampler = reverse_feat(
            sampled_feat, grid, h, w, align_corners=self.align_corners,
            reverse_sampler=getattr(self, 'reverse_sampler', None))
        return reversed_feats

    def reverse_with_inv_grid(self, sampled_feat, grid, h, w):
        """纯 torch 近似版 reverse（逆网格反解 + bilinear），不依赖 qhull。"""
        from .interpolate import reverse_with_inv_grid as _reverse

        return _reverse(sampled_feat, grid, h, w,
                        align_corners=self.align_corners)

    def reverse_grid(self, grid, h, w):
        """求采样网格的逆网格（[-1,1] 坐标），配合 ``F.grid_sample`` 用。"""
        from .interpolate import reverse_grid as _reverse_grid

        return _reverse_grid(grid, h, w, align_corners=self.align_corners)


# 别名：论文与文档中称 Adaptive Resampling (ARS)
ARS = SaliencySampler


# ============================================================================
# 网格复合：把 backbone 逐级相对网格复合成各级绝对坐标场
# ============================================================================

def compute_real_grids(relative_grids: List[Optional[torch.Tensor]],
                       base_feature_shape: Tuple[int, int, int, int],
                       align_corners: bool) -> List[torch.Tensor]:
    """根据逐级相对采样网格，计算每个特征层对应的真实（累积）坐标场。

    backbone 的 ARS 逐级作用：第 i 级的采样建立在前 i-1 级采样后的坐标系上。
    本函数把相对网格链 ``[g1, g2, ...]`` 级联复合，得到每级特征"每个张量位置
    对应原始均匀坐标系的哪个位置"。所有坐标均在 [-1, 1] 归一化坐标系中，
    因此复合结果与特征分辨率无关，可直接供任意分辨率的特征层查询。

    复合技巧：对"坐标场本身"做 ``F.grid_sample``——用相对网格去采样上一级的
    坐标场，即得 ``real_grid_new[r] = real_grid_prev[g[r]]``。

    Args:
        relative_grids: 相对采样网格列表（backbone 逐级输出，可为 None 表示该级无采样）。
        base_feature_shape: 最高分辨率特征的形状 ``(B, C, H, W)``。
        align_corners: 与 ``F.grid_sample`` 一致的坐标对齐约定。

    Returns:
        各级真实坐标场列表，每个 shape ``(B, H_i, W_i, 2)``（x, y 顺序）。
    """
    B, _, H, W = base_feature_shape
    device = next((g.device for g in relative_grids if g is not None), 'cpu')

    if align_corners:
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
    else:
        y_coords = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
        x_coords = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)

    real_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    real_grids = []
    for rel_grid in relative_grids:
        if rel_grid is None:
            real_grids.append(real_grid)
        else:
            real_grid = F.grid_sample(
                real_grid.permute(0, 3, 1, 2),
                rel_grid.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border',
                align_corners=align_corners,
            ).permute(0, 2, 3, 1)
            real_grids.append(real_grid)
    return real_grids
