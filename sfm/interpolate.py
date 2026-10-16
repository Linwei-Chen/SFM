# -*- coding: utf-8 -*-
"""qhull 非均匀上采样（MSAU 第一级，论文 Eq.17/18 三点重心插值）。

这是 FCN style 模型（UPerNet/PSPNet/FCN/CCNet 等非 FPN 结构）推理路径中
MSAU 的非均匀上采样实现：ARX 采样网格 ``grid`` 把特征摊到非均匀坐标上，
``reverse`` 用 Delaunay 三角剖分把非均匀采样结果重建回均匀网格——
对每个均匀网格点，找到其所在三角形的三个顶点，按论文 Eq.17/18 的重心权重
加权求和（原文 "Non-uniform upsampling achieves this through triangulation"）。

代码移植自原始仓库 ``mmseg_custom/models/backbones/interp2d.py``，
数值行为与训练/评测时完全一致（bit-exact）。

qhull 后端选择（按优先级自动探测，可用 :func:`qhull_backend` 查询）：
  1. ``sfm.vendor.spatial.qhull``：随本包携带的 scipy 同源 Cython 源码编译版
     （需要 C 编译器和 LAPACK；``pip install`` 时由 setup.py 自动尝试编译）。
     与原始训练环境用的 qhull 完全同代码，bit-exact。
  2. 顶层 ``qhull`` 模块：兼容用户环境里已有的 vendored spatial。
  3. ``scipy.spatial.Delaunay`` 回退：同一 Qhull 引擎，三角剖分结果一致；
     重心权重用 ``transform`` 矩阵手算（公式同 scipy 文档）。安装时若未能
     编译扩展则自动落到这里，数值与后端 1 的差异见 tests/test_interpolate.py
     （float64 下逐位一致才判定通过；不一致时后端 1 为准）。

纯 torch 的近似替代（``reverse_with_inv_grid``，二次方程反解逆网格 + bilinear
``grid_sample``）也一并携带，供不想依赖 qhull/scipy 的场景使用；它的数值
与 qhull 路径有系统性小差异，仅作工程备选，不用于复现论文数值。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Interp2D', 'invert_nonseparable_grid', 'reverse_feat',
    'reverse_with_inv_grid', 'reverse_grid', 'qhull_backend',
]


# ---------------------------------------------------------------------------
# qhull 后端探测
# ---------------------------------------------------------------------------

def _import_qhull():
    """按优先级加载 qhull 提供者（模块或 Delaunay 工厂）。"""
    try:  # 1) 本包 vendored 编译版
        from .vendor.spatial import qhull as _q
        return _q
    except Exception:
        pass
    try:  # 2) 用户环境顶层 qhull（原训练树 vendored spatial 的用法）
        import qhull as _q
        return _q
    except Exception:
        pass
    try:  # 3) scipy 回退（包装出同接口）
        from scipy.spatial import Delaunay as _ScipyDelaunay
        return _make_scipy_provider(_ScipyDelaunay)
    except Exception:
        return None


def _make_scipy_provider(ScipyDelaunay):
    """把 scipy.spatial.Delaunay 包装成 vendored qhull 的接口。

    vendored qhull 的 ``find_simplex(xi, return_c=True)`` 在 Cython 内部用
    directed search 逐点求重心坐标；scipy 公开 API 无 ``return_c``，这里用
    ``tri.transform``（同一变换矩阵）按公式 c[:2] = A·(x-b)、c[2] = 1-sum
    计算，float64 下与 vendored 版逐位一致（tests 里有三方对拍）。
    """
    class _ScipyQhullModule:
        Delaunay = None

    class _ScipyDelaunayShim:
        """``find_simplex(..., return_c=True)`` 兼容 vendored qhull.Delaunay。"""

        def __init__(self, points, **kwargs):
            self._tri = ScipyDelaunay(np.ascontiguousarray(points,
                                                          dtype=np.float64))
            self.simplices = self._tri.simplices

        def find_simplex(self, xi, bruteforce=False, tol=None,
                         return_c=False):
            xi = np.asanyarray(xi)
            in_shape = xi.shape[:-1]
            xf = np.ascontiguousarray(xi.reshape(-1, xi.shape[-1]),
                                      dtype=np.float64)
            idx = self._tri.find_simplex(xf, bruteforce=bruteforce, tol=tol)
            if not return_c:
                return idx.reshape(in_shape)
            # 重心坐标：transform[n] = [[A(2x2)], [b]]，c[:2]=A@(x-b)。
            # 越界点（idx==-1）在 vendored 版里 c 为未初始化值且随后被
            # Interp2D 以 simplex 0 的顶点乘掉；这里置 0 权重，行为只在
            # 理论上才触发的病理情形与原版不同（正常数据不会出现 -1）。
            flat_idx = idx.reshape(-1)
            ndim = self._tri.points.shape[1]
            safe_idx = np.where(flat_idx >= 0, flat_idx, 0)
            t = self._tri.transform[safe_idx]
            b = np.einsum('nij,nj->ni', t[:, :ndim, :],
                          xf - t[:, ndim, :])
            c = np.concatenate([b, (1.0 - b.sum(axis=1))[:, None]], axis=1)
            c[flat_idx < 0] = 0.0
            return idx.reshape(in_shape), c.reshape(in_shape + (ndim + 1,))

    mod = _ScipyQhullModule()
    mod.Delaunay = _ScipyDelaunayShim
    return mod


_QHULL = _import_qhull()


def qhull_backend():
    """返回当前 qhull 后端名：'vendored' / 'top-level' / 'scipy' / None。"""
    mod = _QHULL
    if mod is None:
        return None
    if getattr(mod, '__name__', '').startswith('sfm.vendor'):
        return 'vendored'
    if getattr(mod, '__name__', '') == 'qhull':
        return 'top-level'
    return 'scipy'


# ---------------------------------------------------------------------------
# Interp2D —— 原版逐行移植（bit-exact）
# ---------------------------------------------------------------------------

class Interp2D(nn.Module):
    """基于 qhull Delaunay 三角剖分的 2D 任意点插值（scipy.griddata 思路）。

    Args:
        h, w: 目标均匀网格的高、宽（插值输出尺寸）。
        add_corner: 是否在三角剖分点集中追加 4 个零值角点，避免边界外
            三角形缺失（原版训练/推理均用默认 ``False``）。

    forward(points, values):
        points: [num, 2] 非均匀采样点坐标（可为 GPU tensor，内部转 CPU 做剖分）。
        values: [num, C]  采样点特征。
        返回:   [C, h, w] 均匀网格上的插值结果。
    """

    def __init__(self, h, w, add_corner=False):
        super().__init__()
        row_coord = np.arange(h).repeat([w]).reshape([h, w])
        col_coord = np.arange(w).repeat([h]).reshape([w, h]).T
        self.coord = np.stack([row_coord, col_coord])
        self.coord = self.coord.transpose([1, 2, 0]).reshape([-1, 2])
        self.add_corner = add_corner
        self.w = w
        self.h = h

    def forward(self, points, values):
        # gradients: the numpy qhull operations find triangular simplices
        # (tri_map --- corner locations) and weights for interpolation;
        # tri_map and weights are not derivable, but it's ok, because the
        # qhull triangulation is deterministic and no parameters are learned.
        # Gradients still flow because values never leave the GPU — we only
        # use tri_map to gather pixels from values.
        if _QHULL is None:
            raise ImportError(
                'qhull 后端不可用：既未能编译 vendored spatial.qhull（需要 C '
                '编译器与 LAPACK），环境中也没有 scipy。请安装 scipy，或用 '
                'reverse_with_inv_grid（纯 torch 近似路径）。')
        if self.add_corner:
            points = torch.cat([points.cpu(), torch.Tensor(
                [[0, 0], [0, self.w - 1], [self.h - 1, 0],
                 [self.h - 1, self.w - 1]]).long()], dim=0)
            values = torch.cat(
                [values, torch.zeros([4, values.shape[1]]).to(values.device)],
                dim=0)
        else:
            points = points.cpu()
        self.tri = _QHULL.Delaunay(points)
        vdim = values.shape[-1]
        isimplex, weights = self.tri.find_simplex(self.coord, return_c=True)
        # Points outside the triangulation get the value -1 (原版行为：
        # 乘 0 后落到 simplex 0，权重仍生效——逐位保留该行为)
        if np.sum(isimplex == -1) > 0:
            print('WARNING: {} Points outside the triangulation get the '
                  'value -1, multiplied by 0\n'.format(
                      np.sum(isimplex == -1)))
            isimplex[isimplex == -1] *= 0
        weights = torch.from_numpy(weights).float().to(values.device)
        isimplex = torch.from_numpy(isimplex).to(values.device)
        isimplex = isimplex.long()
        isimplex = isimplex.reshape([-1, 1])
        # shape: isimplex: [h*w, 1]
        simplices = torch.from_numpy(
            np.ascontiguousarray(self.tri.simplices)).long().to(values.device)
        tri_map = torch.gather(simplices, dim=0,
                               index=isimplex.repeat([1, 3]))
        value_corr = [
            torch.gather(values, dim=0,
                         index=tri_map[:, i].reshape([-1, 1]).repeat(
                             [1, vdim]))
            for i in range(3)]
        value_corr = torch.stack(value_corr)
        weights = weights.transpose(1, 0).unsqueeze(2).repeat([1, 1, vdim])
        out = torch.mul(value_corr, weights).sum(dim=0)
        return out.reshape([self.h, self.w, vdim]).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# reverse —— SaliencySampler 的非均匀上采样（原版逐行移植，去 TensorBoard）
# ---------------------------------------------------------------------------

def reverse_feat(sampled_feat, grid, h, w, align_corners=True,
                 reverse_sampler=None):
    """把 ``sample_with_grid`` 摊到非均匀网格的特征重建回 (h, w) 均匀网格。

    对原始 ``SaliencySampler.reverse`` 的逐行移植（仅去掉 TensorBoard
    可视化与按 batch 循环外的部分不变）。

    Args:
        sampled_feat: [b, C, sf_h, sf_w] 非均匀采样后的特征。
        grid: [b, 2, sf_h, sf_w]（或 [b, sf_h, sf_w, 2]）采样网格，
            取值 [-1, 1]。
        h, w: 目标均匀网格尺寸（一般为原图/上一层特征尺寸）。
        align_corners: 与采样时的坐标约定保持一致。
        reverse_sampler: 可复用的 :class:`Interp2D` 实例（按 h/w 缓存；
            传 None 则新建并返回，便于调用方复用）。

    Returns:
        (reversed_feat [b, C, h, w], reverse_sampler)
    """
    if reverse_sampler is None or reverse_sampler.h != h \
            or reverse_sampler.w != w:
        reverse_sampler = Interp2D(h, w)
    # for grid in b, 2, h, w
    if grid.size(1) == 2 and grid.size(3) != 2:
        grid = grid.permute(0, 2, 3, 1)
    grid = grid.clone().contiguous()
    if align_corners:
        grid[:, :, :, 0] = 0.5 * (grid[:, :, :, 0] + 1) * (w - 1)
        grid[:, :, :, 1] = 0.5 * (grid[:, :, :, 1] + 1) * (h - 1)
    else:
        grid[:, :, :, 0] = 0.5 * (grid[:, :, :, 0] + 1) - 0.5 / w
        grid[:, :, :, 1] = 0.5 * (grid[:, :, :, 1] + 1) - 0.5 / h
        grid[:, :, :, 0] = grid[:, :, :, 0] * w
        grid[:, :, :, 1] = grid[:, :, :, 1] * h
        grid[:, :, 0, 0] = 0
        grid[:, :, -1, 0] = w - 1
        grid[:, 0, :, 1] = 0
        grid[:, -1, :, 1] = h - 1
    b, _, sf_h, sf_w = sampled_feat.shape
    grid = grid.view(b, sf_h * sf_w, 2)
    grid = grid[:, :, [1, 0]]
    sampled_feat = sampled_feat.permute(0, 2, 3, 1).view(b, sf_h * sf_w, -1)

    reversed_feats = []
    for i in range(b):
        reversed_feat = reverse_sampler(grid[i].detach(), sampled_feat[i])
        reversed_feats.append(reversed_feat)
    reversed_feats = torch.stack(reversed_feats, dim=0)
    return reversed_feats, reverse_sampler


def reverse_grid(grid, h, w, align_corners=True):
    """求采样网格的逆网格（[-1,1] 坐标），配合 ``F.grid_sample`` 使用。"""
    inv_grid = invert_nonseparable_grid(grid.permute(0, 2, 3, 1), h, w,
                                        align_corners=align_corners)
    return inv_grid.permute(0, 3, 1, 2)


def reverse_with_inv_grid(sampled_feat, grid, h, w, align_corners=True):
    """纯 torch 的近似 reverse：二次方程反解逆网格 + bilinear grid_sample。

    与 qhull 路径（:func:`reverse_feat`）数值存在系统性小差异，仅作不依赖
    qhull/scipy 时的工程备选，不用于复现论文数值。
    """
    inv_grid = invert_nonseparable_grid(grid.permute(0, 2, 3, 1), h, w,
                                        align_corners=align_corners)
    reversed_feat = F.grid_sample(sampled_feat, inv_grid, mode='bilinear',
                                  align_corners=align_corners)
    return reversed_feat


def invert_nonseparable_grid(grid, H, W, align_corners=True):
    """逐 cell 二次方程反解，把 [-1,1] 前向网格映回均匀目标网格的逆映射。

    原版逐行移植（SaliencySampler 使用的 ``invert_nonseparable_grid``）。
    """
    from math import ceil, floor

    grid = grid.clone()
    device = grid.device
    B, grid_H, grid_W, _ = grid.shape  # B, H, W, 2, in (-1, 1), xy
    eps = 1e-8
    if align_corners:
        grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
        grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    else:
        # grid in [+1/2H, H-1/2H]
        grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * W
        grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * H
    # grid now ranges from 0 to ([H or W] - 1)
    inverse_grid = 2 * max(H, W) * torch.ones(
        (B, H, W, 2), dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b, :-1, :-1, :].contiguous().view(-1, 2)
        p10 = grid[b, 1:, :-1, :].contiguous().view(-1, 2)
        p01 = grid[b, :-1, 1:, :].contiguous().view(-1, 2)
        p11 = grid[b, 1:, 1:, :].contiguous().view(-1, 2)

        ref = torch.floor(p00).type(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        v11 = p11 - ref

        min_x = int(floor(min(v00[:, 0].min(), v10[:, 0].min()) - eps))
        max_x = int(ceil(max(v01[:, 0].max(), v11[:, 0].max()) + eps))
        min_y = int(floor(min(v00[:, 1].min(), v01[:, 1].min()) - eps))
        max_y = int(ceil(max(v10[:, 1].max(), v11[:, 1].max()) + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T

        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        vb = v10 - v00
        vc = v01 - v00
        vd = v00 - v10 - v01 + v11

        vx = pts.permute(1, 0).unsqueeze(0)  # 1 x (x_range*y_range) x 2
        Ma = v00.unsqueeze(1) - vx  # ((grid_H-1)*(grid_W-1)) x (xr*yr) x 2

        vc_cross_vd = (vc[:, 0] * vd[:, 1]
                       - vc[:, 1] * vd[:, 0]).unsqueeze(1)
        vc_cross_vb = (vc[:, 0] * vb[:, 1]
                       - vc[:, 1] * vb[:, 0]).unsqueeze(1)
        Ma_cross_vd = (Ma[:, :, 0] * vd[:, 1].unsqueeze(1)
                       - Ma[:, :, 1] * vd[:, 0].unsqueeze(1))
        Ma_cross_vb = (Ma[:, :, 0] * vb[:, 1].unsqueeze(1)
                       - Ma[:, :, 1] * vb[:, 0].unsqueeze(1))

        qf_a = vc_cross_vd.expand(*Ma_cross_vd.shape)
        qf_b = vc_cross_vb + Ma_cross_vd
        qf_c = Ma_cross_vb

        mu_neg = -1 * torch.ones_like(Ma_cross_vd)
        mu_pos = -1 * torch.ones_like(Ma_cross_vd)
        mu_linear = -1 * torch.ones_like(Ma_cross_vd)

        nzie = (qf_a.abs() > 1e-10).expand(*Ma_cross_vd.shape)

        disc = (qf_b[nzie] ** 2 - 4 * qf_a[nzie] * qf_c[nzie]) ** 0.5
        mu_pos[nzie] = (-qf_b[nzie] + disc) / (2 * qf_a[nzie])
        mu_neg[nzie] = (-qf_b[nzie] - disc) / (2 * qf_a[nzie])
        mu_linear[~nzie] = qf_c[~nzie] / qf_b[~nzie]

        mu_pos_valid = torch.logical_and(mu_pos >= 0, mu_pos <= 1)
        mu_neg_valid = torch.logical_and(mu_neg >= 0, mu_neg <= 1)
        mu_linear_valid = torch.logical_and(mu_linear >= 0, mu_linear <= 1)

        mu = -1 * torch.ones_like(Ma_cross_vd)
        mu[mu_pos_valid] = mu_pos[mu_pos_valid]
        mu[mu_neg_valid] = mu_neg[mu_neg_valid]
        mu[mu_linear_valid] = mu_linear[mu_linear_valid]

        lmbda = -1 * (Ma[:, :, 1] + mu * vc[:, 1:2]) / (
            vb[:, 1:2] + vd[:, 1:2] * mu)

        unwarped_pts = torch.stack((lmbda, mu), dim=0)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1 + eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1 + eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + \
            ref[nonzero_good_indices[:, 0], 0]
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + \
            ref[nonzero_good_indices[:, 0], 1]
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices],
             i + unwarped_pts[0, good_indices]),
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = \
            grid_mappings[in_bounds, :]
    if align_corners:
        inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0
        inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0
    else:
        inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W) * 2.0 - 1.0
        inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H) * 2.0 - 1.0
    clamp_value_x = 1 if align_corners else (1 - 0.5 / W * 2)
    clamp_value_y = 1 if align_corners else (1 - 0.5 / H * 2)
    inverse_grid[..., 0] = torch.clamp(inverse_grid[..., 0],
                                       min=-clamp_value_x,
                                       max=clamp_value_x)
    inverse_grid[..., 1] = torch.clamp(inverse_grid[..., 1],
                                       min=-clamp_value_y,
                                       max=clamp_value_y)
    return inverse_grid.detach()
