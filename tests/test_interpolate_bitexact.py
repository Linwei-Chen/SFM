#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三方 bit-exact 对拍：原版 Interp2D（原文件源码逐字节执行）vs sfm 移植版 vs scipy 回退。

用真实服务器路径在 3090-8 上运行：
    /data4/chenlinwei/anaconda3/envs/mmseg/bin/python tests/test_interpolate_bitexact.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, '/data4/chenlinwei/code/sfm_eval_dcn')

from sfm import interpolate
from sfm.interpolate import Interp2D, _make_scipy_provider

ORIG_INTERP2D_FILE = ('/data4/chenlinwei/code/ResolutionDet/InternImage/'
                      'segmentation/mmseg_custom/models/backbones/interp2d.py')


def load_original_intero2d():
    """从原文件逐字节取出 Interp2D 类源码并执行（注入同一个 vendored qhull）。"""
    src = open(ORIG_INTERP2D_FILE).read()
    start = src.index('class Interp2D')
    end = src.index('def test_1')
    import sfm.vendor.spatial.qhull as vendored
    ns = {'np': np, 'torch': torch, 'nn': torch.nn, 'qhull': vendored}
    exec(compile(src[start:end], ORIG_INTERP2D_FILE, 'exec'), ns)
    return ns['Interp2D']


def make_case(h, w, n, c, seed):
    """构造 ARS 风格用例：点覆盖四角（边界钉住）+ 内部扰动点，(row, col) 顺序。"""
    g = torch.Generator().manual_seed(seed)
    points = torch.rand(n, 2, generator=g) * torch.tensor([h - 1, w - 1]).float()
    points[0] = torch.tensor([0.0, 0.0])
    points[1] = torch.tensor([float(w - 1), 0.0])
    points[2] = torch.tensor([0.0, float(h - 1)])
    points[3] = torch.tensor([float(w - 1), float(h - 1)])
    values = torch.randn(n, c, generator=g)
    return points, values


def main():
    print('qhull backend:', interpolate.qhull_backend())
    OrigInterp2D = load_original_intero2d()
    scipy_provider = _make_scipy_provider(
        __import__('scipy.spatial', fromlist=['Delaunay']).Delaunay)

    # 覆盖：小/中/大尺寸、多通道、奇数尺寸（33x65 = stride-32 反解的真实形状）
    cases = [
        (10, 10, 100, 1, 0),
        (64, 64, 256, 8, 1),
        (128, 64, 1024, 16, 2),
        (33, 65, 2145, 64, 3),
        (65, 129, 8385, 32, 4),
    ]
    n_fail = 0
    for h, w, n, c, seed in cases:
        pts, vals = make_case(h, w, n, c, seed)
        out_sfm = Interp2D(h, w)(pts, vals)
        out_orig = OrigInterp2D(h, w)(pts, vals)
        saved = interpolate._QHULL
        interpolate._QHULL = scipy_provider
        try:
            out_scipy = Interp2D(h, w)(pts, vals)
        finally:
            interpolate._QHULL = saved
        eq_orig = torch.equal(out_sfm, out_orig)
        eq_scipy = torch.equal(out_sfm, out_scipy)
        d_orig = (out_sfm - out_orig).abs().max().item()
        d_scipy = (out_sfm - out_scipy).abs().max().item()
        print(f'case h={h:4d} w={w:4d} n={n:6d} c={c:3d} | '
              f'sfm==原版 bit-exact={eq_orig} (max|d|={d_orig:.3e}) | '
              f'sfm==scipy bit-exact={eq_scipy} (max|d|={d_scipy:.3e})')
        if not eq_orig:
            n_fail += 1

    # ---- 真实 ARS 网格（边界钉住、凸包全覆盖 → 无 outside 点）----
    # 同时覆盖 reverse 的坐标换算（align_corners=False 分支 + xy 交换）
    from sfm import ARS, swap_syncbn_to_bn
    torch.manual_seed(0)
    for hw in [(32, 64), (16, 33)]:
        ss = swap_syncbn_to_bn(ARS(
            feat_dim=8, stride=1, padding_size=5, fwhm=3,
            sampler_mode='avg_lhpf_psp', psp_ratio=4,
            align_corners=False, psp_bins=[1, 2, 3, 7])).cuda()
        feat = torch.randn(2, 8, *hw).cuda()
        _, feat_s, sal, grid = ss(feat, sample_with_grid=True)
        out_sfm = ss.reverse(feat_s, grid, feat.shape[2], feat.shape[3])
        saved = interpolate._QHULL
        interpolate._QHULL = scipy_provider
        try:
            ss2 = swap_syncbn_to_bn(ARS(
                feat_dim=8, stride=1, padding_size=5, fwhm=3,
                sampler_mode='avg_lhpf_psp', psp_ratio=4,
                align_corners=False, psp_bins=[1, 2, 3, 7])).cuda()
            ss2.load_state_dict(ss.state_dict())
            _, feat_s2, _, grid2 = ss2(feat, sample_with_grid=True)
            out_scipy, _ = interpolate.reverse_feat(
                feat_s2, grid2, feat.shape[2], feat.shape[3],
                align_corners=False)
        finally:
            interpolate._QHULL = saved
        eq = torch.equal(out_sfm, out_scipy)
        d = (out_sfm - out_scipy).abs().max().item()
        print(f'real-ARS grid {hw} | sfm.reverse==scipy bit-exact={eq} '
              f'(max|d|={d:.3e})')
        if not eq:
            n_fail += 1

    print('结论：' + ('全部通过' if n_fail == 0 else f'{n_fail} 个用例失败'))
    return n_fail


if __name__ == '__main__':
    sys.exit(main())
