# -*- coding: utf-8 -*-
"""SFM_Align 单元测试（纯 CPU、零框架依赖，pytest 或直接 python 运行均可）。

数值金标准：
1. ``carafe_reassembly`` vs 逐像素参考实现（本文件内的双循环）；
2. 在装有 mmcv 的环境中可与 ``mmcv.ops.carafe`` 对照（见 tests/README 说明，
   已在 mmcv 1.5.3 + CUDA 上验证 max_diff ≈ 2.4e-07）。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SFM_Align import (AlignModule, LPRMAlignUpModule, LPRM_efficient,
                       LocalPixelRelationConv, _neighbor_reassembly,
                       carafe_reassembly, compute_real_grids)

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


def _swap_syncbn_to_bn(module):
    """把 SyncBatchNorm 替换为 BatchNorm2d（eval 数学等价、state_dict 同构），
    使 CPU 单机测试可跑；GPU/DDP 训练时模块内仍是 SyncBN。"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SyncBatchNorm):
            bn = torch.nn.BatchNorm2d(child.num_features)
            bn.load_state_dict(child.state_dict())
            setattr(module, name, bn)
        else:
            _swap_syncbn_to_bn(child)
    return module


def test_local_pixel_relation_conv_math():
    """LocalPixelRelationConv 的邻域聚合与手写双循环一致（权重取自 weight_net）。"""
    torch.manual_seed(1)
    B, C, H, W, k, d = 1, 4, 5, 5, 3, 2
    lprc = LocalPixelRelationConv(C, kernel_size=k, dilation=d,
                                  use_adaptive_residual=False, use_channel=False)
    lprc = _swap_syncbn_to_bn(lprc)
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


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'{fn.__name__} ... ok')
    print(f'\n{len(fns)} tests passed.')
