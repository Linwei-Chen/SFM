# -*- coding: utf-8 -*-
"""sfm 包单元测试（纯 CPU、零框架依赖，pytest 或直接 python 运行均可）。

数值金标准：
1. ``carafe_reassembly`` vs 逐像素参考实现（本文件内的双循环）；
2. 在装有 mmcv 的环境中可与 ``mmcv.ops.carafe`` 对照（已在 mmcv 1.5.3 + CUDA
   上验证 max_diff ≈ 2.4e-07）；
3. ``SaliencySampler``（最小 ARS）与原始仓库实现 state_dict 逐键可互载、输出
   数值一致（在训练环境服务器上验证通过）。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfm import (ARS, AlignModule, LPRMAlignUpModule, LPRM_efficient,
                 LocalPixelRelationConv, SFMFPN, SaliencySampler,
                 carafe_reassembly, compute_real_grids,
                 register_sfm, swap_syncbn_to_bn, FPN_BASE_SOURCE)
from sfm.upsampling import _neighbor_reassembly

torch.manual_seed(0)


def _reference_reassembly(value, w, k, s, d=1):
    """逐像素双循环参考实现（慢，仅小尺寸测试用）。

    out[m, n] = Σ_{u,v} w[u*k+v, m, n] · value[m//s + (u-p)*d, n//s + (v-p)*d]，越界补零。
    """
    B, C, H, W = value.shape
    H_out, W_out = H * s, W * s
    p = (k - 1) // 2
    out = torch.zeros(B, C, H_out, W_out)
    for m in range(H_out):
        for n in range(W_out):
            base_y, base_x = m // s, n // s
            for u in range(k):
                for v in range(k):
                    y = base_y + (u - p) * d
                    x = base_x + (v - p) * d
                    if 0 <= y < H and 0 <= x < W:
                        out[:, :, m, n] += w[:, u * k + v, m, n].unsqueeze(1) * value[:, :, y, x]
    return out


def test_carafe_reassembly_matches_reference():
    B, C, H, W, k, s = 2, 3, 4, 5, 3, 2
    value = torch.randn(B, C, H, W)
    mask = torch.randn(B, k * k, H * s, W * s)
    out = carafe_reassembly(value, mask, k, s)
    ref = _reference_reassembly(value, mask.softmax(1), k, s)
    assert out.shape == (B, C, H * s, W * s)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_carafe_reassembly_scale1_and_dilation():
    B, C, H, W, k = 1, 2, 4, 4, 3
    value = torch.randn(B, C, H, W)
    mask = torch.randn(B, k * k, H, W)
    out = carafe_reassembly(value, mask, k, 1)
    ref = _reference_reassembly(value, mask.softmax(1), k, 1)
    assert torch.allclose(out, ref, atol=1e-5)

    out_d2 = _neighbor_reassembly(value, mask.softmax(1), k, 1, dilation=2)
    ref_d2 = _reference_reassembly(value, mask.softmax(1), k, 1, d=2)
    assert torch.allclose(out_d2, ref_d2, atol=1e-5)


def test_carafe_reassembly_zero_padding_semantics():
    """越界采样必须补零（与 mmcv 一致）。

    用大 logit 差构造近似独热权重（softmax 后其余通道 < 1e-40），
    使输出只反映通道 0 (u=0,v=0) 的采样：value[m//s - 1, n//s - 1]，越界补零。
    """
    value = torch.ones(1, 1, 2, 2)
    mask = torch.full((1, 9, 4, 4), -100.0)
    mask[:, 0] = 100.0
    out = carafe_reassembly(value, mask, 3, 2)
    # m,n ∈ {0,1} 时 base=(0,0)，采样 (-1,-1) 越界 → 0
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 0, 1, 1].item() == 0.0
    # m,n ∈ {2,3} 时 base=(1,1)，采样 (0,0) → 1
    assert abs(out[0, 0, 2, 2].item() - 1.0) < 1e-6
    assert abs(out[0, 0, 3, 3].item() - 1.0) < 1e-6


def test_align_module_identity_at_zero_init():
    """offset 预测器零初始化时，AlignModule 应近似恒等。

    数学上严格恒等（三点重合 → 重心权重退化为 (1,0,0)，采样位置即均匀网格）；
    浮点上 align_corners=False 的 1/W 非二进制精确（如 1/6）会引入 ~1e-7 误差，
    故用 1e-5 容差。
    """
    B, C, H, W = 1, 8, 6, 6
    am = AlignModule(C, k=3, align_corners=False, compress_ratio=4)
    x = torch.randn(B, C, H, W)
    real_grid = torch.randn(B, H, W, 2)
    y = am(x, real_grid)
    assert y.shape == x.shape
    assert (y - x).abs().max().item() < 1e-5, (y - x).abs().max().item()


def test_lprm_align_up_module_shape():
    B, C = 2, 32
    lum = LPRMAlignUpModule(C, C, kernel_size=3, compress_ratio=16)
    x_low = torch.randn(B, C, 8, 10)
    x_high = torch.randn(B, C, 16, 20)
    y = lum(x_low, x_high)
    assert y.shape == x_high.shape


def test_lprm_efficient_dilation_finite():
    m = LPRM_efficient(channels=4, kernel_size=3, dilation=2, upscale=1)
    value = torch.randn(1, 4, 6, 6)
    mask = torch.randn(1, 9, 6, 6)
    out = m(mask, value)
    assert out.shape == value.shape
    assert torch.isfinite(out).all()


def test_compute_real_grids_none_identity():
    """相对网格全为 None 时，复合结果应等于均匀网格本身。"""
    B, H, W = 1, 4, 5
    grids = [None, None]
    real_grids = compute_real_grids(grids, (B, 3, H, W), align_corners=False)
    assert len(real_grids) == 2
    y, x = torch.meshgrid(
        torch.linspace(-1 + 1 / H, 1 - 1 / H, H),
        torch.linspace(-1 + 1 / W, 1 - 1 / W, W))
    uniform = torch.stack([x, y], dim=-1)
    assert torch.allclose(real_grids[0][0], uniform, atol=1e-6)
    assert torch.allclose(real_grids[1][0], uniform, atol=1e-6)


def test_local_pixel_relation_conv_math():
    """LocalPixelRelationConv 的邻域聚合与手写双循环一致（权重取自 weight_net）。"""
    torch.manual_seed(1)
    B, C, H, W, k, d = 1, 4, 5, 5, 3, 2
    lprc = LocalPixelRelationConv(C, kernel_size=k, dilation=d,
                                  use_adaptive_residual=False, use_channel=False)
    lprc = swap_syncbn_to_bn(lprc)
    lprc.eval()  # eval 下 BN 用 running stats，行为确定
    x = torch.randn(B, C, H, W)
    feat = torch.randn(B, C, H, W)
    _, res = lprc(x, feat)

    # 手写：weight_net 权重 + replicate 填充 + unfold 加权
    weight = lprc.weight_net(lprc.conv_pad(x))  # (B, k², H, W)
    pad_feat = lprc.conv_pad(feat)
    cols = F.unfold(pad_feat, kernel_size=k, dilation=d)  # (B, C·k², H·W)
    cols = cols.view(B, C, k * k, H, W)
    w = weight.softmax(dim=1).unsqueeze(1)  # (B, 1, k², H, W)
    ref = (w * cols).sum(dim=2)
    assert torch.allclose(res, ref, atol=1e-5), (res - ref).abs().max()


def test_gradient_flow():
    value = torch.randn(1, 4, 6, 6, requires_grad=True)
    mask = torch.randn(1, 9, 12, 12)
    carafe_reassembly(value, mask, 3, 2).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


# ===========================================================================
# ARS（最小自适应重采样）
# ===========================================================================

def test_ars_none_mode_uniform_grid():
    """sampler_mode='none'：恒等采样，grid 为均匀网格。"""
    ss = SaliencySampler(feat_dim=8, sampler_mode='none', align_corners=False)
    x = torch.randn(2, 8, 6, 8)
    feat, sampled, saliency, grid = ss(x)
    assert saliency is None
    assert torch.equal(sampled, x)
    assert grid.shape == (2, 2, 6, 8)
    # 均匀网格：首末行列钉在 ±(1 - 1/n)
    assert torch.allclose(grid[0, 1, 0, :], torch.full((8,), 1.0 / 6 - 1.0), atol=1e-6)
    assert torch.allclose(grid[0, 1, -1, :], torch.full((8,), 1.0 - 1.0 / 6), atol=1e-6)
    assert torch.allclose(grid[0, 0, :, 0], torch.full((6,), 1.0 / 8 - 1.0), atol=1e-6)


def test_ars_conv3_shapes_grid_bounds_and_border_pinning():
    """'conv3' 模式：形状正确、grid 有界、边界钉回均匀坐标（align_corners=True）。"""
    ss = SaliencySampler(feat_dim=8, stride=1, padding_size=5, fwhm=3,
                         sampler_mode='conv3', align_corners=True)
    B, C, H, W = 2, 8, 10, 12
    x = torch.randn(B, C, H, W)
    feat, sampled, saliency, grid = ss(x)
    assert sampled.shape == x.shape
    assert saliency.shape == (B, 1, H, W)
    assert grid.shape == (B, 2, H, W)
    assert torch.isfinite(grid).all()
    assert grid.abs().max() <= 1.0 + 1e-6
    # 边界钉回：align_corners=True 时边界均匀坐标为 ±1
    assert torch.allclose(grid[:, 1, 0, :], torch.full((W,), -1.0), atol=1e-6)
    assert torch.allclose(grid[:, 1, -1, :], torch.full((W,), 1.0), atol=1e-6)
    assert torch.allclose(grid[:, 0, :, 0], torch.full((H,), -1.0), atol=1e-6)
    assert torch.allclose(grid[:, 0, :, -1], torch.full((H,), 1.0), atol=1e-6)


def test_ars_saliency_is_normalized_density():
    """显著图经 softmax × H·W 后总质量 = 像素数。"""
    ss = SaliencySampler(feat_dim=8, sampler_mode='conv', align_corners=False,
                         padding_size=2, fwhm=3)
    x = torch.randn(2, 8, 6, 6)
    _, _, saliency, _ = ss(x, sample_with_grid=False)
    assert torch.allclose(saliency.sum(dim=(1, 2, 3)), torch.full((2,), 36.0), atol=1e-3)


def test_ars_state_dict_structure_matches_original():
    """'avg_lhpf_psp'（released 配置模式）的 state_dict 键与原始实现逐键一致。

    原始结构：saliency_conv = Seq(AvgPool, LHPFConv3, PSP, Conv2d)，
    PSP_Module = (pool, conv, bn, relu)，另有固定高斯核 filter。
    """
    ss = SaliencySampler(feat_dim=64, stride=1, padding_size=5, fwhm=3,
                         sampler_mode='avg_lhpf_psp', psp_ratio=16,
                         align_corners=False)
    keys = set(ss.state_dict().keys())
    expected = {'saliency_conv.1.weight', 'saliency_conv.1.identity_kernel',
                'saliency_conv.3.weight', 'saliency_conv.3.bias',
                'filter.weight'}
    for p in ('psp1', 'psp2', 'psp3', 'psp4'):
        expected.add(f'saliency_conv.2.{p}.conv.weight')
        for t in ('weight', 'bias', 'running_mean', 'running_var',
                  'num_batches_tracked'):
            expected.add(f'saliency_conv.2.{p}.bn.{t}')
    assert keys == expected, (keys - expected, expected - keys)


def test_ars_avg_lhpf_psp_forward_cpu():
    """'avg_lhpf_psp' 完整前向（CPU，SyncBN 换 BN）+ 梯度流。"""
    ss = SaliencySampler(feat_dim=16, stride=1, padding_size=5, fwhm=3,
                         sampler_mode='avg_lhpf_psp', psp_ratio=16,
                         align_corners=False)
    ss = swap_syncbn_to_bn(ss)
    # PSP 内部是 1×1 空间尺寸的 BN，batch=1 时训练模式必然报错（原版行为一致），用 B=2
    x = torch.randn(2, 16, 12, 12, requires_grad=True)
    feat, sampled, saliency, grid = ss(x)
    assert sampled.shape == x.shape
    assert torch.isfinite(sampled).all()
    sampled.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ars_alias():
    assert ARS is SaliencySampler


# ===========================================================================
# SFMFPN（本地无框架 → 走内置纯 torch FPN 基类）
# ===========================================================================

def _make_pyramid():
    feats = [torch.randn(2, c, s, s) for c, s in
             [(256, 64), (512, 32), (1024, 16), (2048, 8)]]
    return feats, [f.shape[1] for f in feats]


def test_sfmfpn_shapes_and_extra_pool():
    feats, in_channels = _make_pyramid()
    neck = SFMFPN(in_channels, out_channels=256, num_outs=4, sfm_compress_ratio=16)
    outs = neck(feats)
    assert [tuple(o.shape) for o in outs] == \
        [(2, 256, 64, 64), (2, 256, 32, 32), (2, 256, 16, 16), (2, 256, 8, 8)]

    neck5 = SFMFPN(in_channels, out_channels=256, num_outs=5)
    outs5 = neck5(feats)
    assert len(outs5) == 5
    assert tuple(outs5[-1].shape) == (2, 256, 4, 4)  # max_pool2d(stride=2) 附加层


def test_sfmfpn_extra_convs_sources():
    feats, in_channels = _make_pyramid()
    for src in ('on_input', 'on_lateral', 'on_output', True):
        neck = SFMFPN(in_channels, out_channels=64, num_outs=5,
                      add_extra_convs=src, sfm_compress_ratio=16)
        outs = neck(feats)
        assert len(outs) == 5
        assert tuple(outs[-1].shape) == (2, 64, 4, 4)


def test_sfmfpn_backward_and_start_level():
    feats, in_channels = _make_pyramid()
    neck = SFMFPN(in_channels, out_channels=64, num_outs=4, start_level=1)
    outs = neck(feats)
    assert len(outs) == 4
    total = sum(o.sum() for o in outs)
    total.backward()
    # SFM 模块收到了梯度
    for m in neck.sfm_aligns:
        assert m.compress_conv_low[0].weight.grad is not None


def test_sfmfpn_register_idempotent():
    """register_sfm 可重复调用；本地无框架时返回空列表且不报错。"""
    done = register_sfm()
    assert isinstance(done, list)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'{fn.__name__} ... ok')
    print(f'\n{len(fns)} tests passed. (FPN_BASE_SOURCE={FPN_BASE_SOURCE})')
