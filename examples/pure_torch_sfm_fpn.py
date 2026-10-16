# -*- coding: utf-8 -*-
"""纯 PyTorch 用法（无需任何框架）：pip install sfm 后直接跑通本文件。

    python examples/pure_torch_sfm_fpn.py
"""
import sys
from pathlib import Path

import torch

try:
    from sfm import ARS, LPRMAlignUpModule, SFMFPN
except ModuleNotFoundError:  # 未 pip install 时回退到仓库根目录的源码包
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sfm import ARS, LPRMAlignUpModule, SFMFPN

torch.manual_seed(0)

# ---- 1) FPN 层间融合：SFMFPN（与官方 FPN 同签名） ----
feats = [torch.randn(2, c, s, s)
         for c, s in [(256, 64), (512, 32), (1024, 16), (2048, 8)]]
neck = SFMFPN([256, 512, 1024, 2048], out_channels=256, num_outs=5,
              sfm_compress_ratio=16)
outs = neck(feats)
print('SFMFPN 输出:', [tuple(o.shape) for o in outs])

# ---- 2) 张量级极简用法：两个特征层之间的对齐上采样 ----
hr = torch.randn(2, 256, 64, 64)   # 浅层（分辨率高）
lr = torch.randn(2, 256, 32, 32)   # 深层（语义强）
align_up = LPRMAlignUpModule(256, 256, kernel_size=3, compress_ratio=16)
hr_fused = hr + align_up(x_low=lr, guidance_high_aligned=hr)
print('层间融合:', tuple(hr_fused.shape))

# ---- 3) backbone 下采样处接入 ARS（自适应重采样） ----
from sfm import swap_syncbn_to_bn

ss = ARS(feat_dim=256, stride=1, padding_size=5, fwhm=3,
         sampler_mode='avg_lhpf_psp', psp_ratio=16, align_corners=False)
# ARS 的显著图预测内含 SyncBN（与 released checkpoint 逐键一致，为 DDP 训练设计）；
# 单机/推理场景先换成等价的普通 BN（state_dict 同构，加载权重前后均可调用）
ss = swap_syncbn_to_bn(ss).eval()
x = torch.randn(2, 256, 32, 32)
with torch.no_grad():
    x, x_sampled, saliency, grid = ss(x, sample_with_grid=True)
print('ARS:', 'saliency', tuple(saliency.shape), '| grid', tuple(grid.shape),
      '| 采样后特征', tuple(x_sampled.shape))
# x_sampled 交给随后的 stage 卷积正常降采样；grid 是"揉形记录"，
# 供上游 AlignModule + compute_real_grids 摊平使用
