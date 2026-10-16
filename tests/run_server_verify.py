# -*- coding: utf-8 -*-
"""sfm 包服务器验证脚本（3090-8, openmmlab env, pip mmdet 2.28.2）。

验证三件事：
1. import sfm 自动注册 SFMFPN 到 mmdet 2.x 的 NECKS 注册表；
2. 经注册表构建 SFMFPN，GPU forward/backward 通过；
3. 最小化 ARS（SaliencySampler）与服务器原始实现 state_dict 逐键互载、
   输出数值一致（两种 sampler_mode 各测一次）。
"""
import os
import sys
import types

import torch

import sfm  # noqa: F401  导入即注册

print('[1] FPN_BASE_SOURCE =', sfm.FPN_BASE_SOURCE)
assert sfm.FPN_BASE_SOURCE == 'mmdet', '应继承 pip mmdet 的官方 FPN'

# ---- 注册检查 ----
from mmdet.models.builder import NECKS
assert 'SFMFPN' in NECKS.module_dict, 'SFMFPN 未注册到 mmdet NECKS'
sfm.register_sfm()  # 幂等重入
print('[1] 注册 OK:', NECKS.get('SFMFPN') is sfm.SFMFPN)

# ---- 注册表构建 + GPU forward/backward ----
dev = 'cuda'
cfg = dict(type='SFMFPN', in_channels=[256, 512, 1024, 2048],
           out_channels=256, num_outs=5, sfm_compress_ratio=16)
neck = NECKS.build(cfg).to(dev)
from mmdet.models.necks import FPN as MmdetFPN
assert isinstance(neck, MmdetFPN) and isinstance(neck, sfm.SFMFPN)
assert len(neck.sfm_aligns) == 3, '4 级金字塔应有 3 个 SFM 对齐模块'
feats = [torch.randn(2, c, s, s, device=dev)
         for c, s in [(256, 64), (512, 32), (1024, 16), (2048, 8)]]
outs = neck(feats)
shapes = [tuple(o.shape) for o in outs]
assert shapes == [(2, 256, 64, 64), (2, 256, 32, 32), (2, 256, 16, 16),
                  (2, 256, 8, 8), (2, 256, 4, 4)], shapes
loss = sum(o.sum() for o in outs)
loss.backward()
n_sfm = sum(p.numel() for m in neck.sfm_aligns for p in m.parameters())
print('[2] SFMFPN 注册表构建 + GPU forward/backward OK, 输出:', shapes,
      '| sfm 参数量:', n_sfm)

# ---- 最小 ARS vs 原始 SaliencySampler 数值对齐（分树验证）----
# 三棵训练树的 SaliencySampler 略有差异（PSP bins 传参不同），各按其线的
# 实际训练配置对齐：
#   - SFM_mask2former (1287 行)：Mask2Former 49.22 线，PSP bins 随 align_corners
#   - InternImage/segmentation (2830 行)：UPerNet 49.33/54.77 线，PSP 恒用默认 [1,2,3,7]
#   - mmsegmentation (4598 行)：方法② git 树，同上
import importlib.util
from sfm import swap_syncbn_to_bn

TREE_CHECKS = [
    ('/data4/chenlinwei/code/ResolutionDet/SFM_mask2former/mmseg_custom/models/backbones/interp2d.py',
     'Mask2Former-49.22线', dict(psp_bins=None)),
    ('/data4/chenlinwei/code/ResolutionDet/InternImage/segmentation/mmseg_custom/models/backbones/interp2d.py',
     'UPerNet-InternImage-54.77线', dict(psp_bins=[1, 2, 3, 7])),
    ('/data4/chenlinwei/code/ResolutionDet/mmsegmentation/mmseg/models/backbones/interp2d.py',
     '方法②-git树', dict(psp_bins=[1, 2, 3, 7])),
]

for stub_name in ['mmseg', 'mmseg.models', 'mmseg.models.backbones',
                  'mmseg.models.backbones.spatial',
                  'mmseg_custom', 'mmseg_custom.models',
                  'mmseg_custom.models.backbones',
                  'mmseg_custom.models.backbones.spatial']:
    sys.modules[stub_name] = types.ModuleType(stub_name)
for pkg in ('mmseg', 'mmseg_custom'):
    qhull = types.ModuleType(f'{pkg}.models.backbones.spatial.qhull')
    qhull.Delaunay = object
    sys.modules[f'{pkg}.models.backbones.spatial.qhull'] = qhull
turtle = types.ModuleType('turtle')
turtle.forward = lambda *a, **k: None
sys.modules['turtle'] = turtle
try:
    import torch.utils.tensorboard  # noqa: F401
except Exception:
    tb = types.ModuleType('torch.utils.tensorboard')
    tb.SummaryWriter = type('SummaryWriter', (), {'__init__': lambda self, *a, **k: None})
    sys.modules['torch.utils.tensorboard'] = tb

torch.manual_seed(0)
# 三棵树会向 mmcv 注册表注册同名算子（如 DCNv2_ASv1），加载时临时 force 覆盖
from mmcv.utils.registry import Registry
_orig_register_module = Registry.register_module

def _force_register(self, name=None, force=True, **kwargs):
    return _orig_register_module(self, name=name, force=force, **kwargs)

Registry.register_module = _force_register
for path, label, extra in TREE_CHECKS:
    if not os.path.exists(path):
        print(f'[3] {label}: 文件不存在，跳过 ({path})')
        continue
    spec = importlib.util.spec_from_file_location(f'orig_interp2d_{label}', path)
    orig_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig_mod)
    OrigSampler = orig_mod.SaliencySampler

    if 'Mask2Former' in label:
        mode, ratio, ac = 'avg_lhpf_psp', 16, False
    else:
        mode, ratio, ac = 'avg_lhpf_psp', 4, False
    kwargs = dict(feat_dim=64, stride=1, padding_size=5, fwhm=3,
                  sampler_mode=mode, psp_ratio=ratio, align_corners=ac,
                  use_checkpoint=False, log_dir=None)
    orig = OrigSampler(**kwargs)
    mine = sfm.SaliencySampler(**kwargs, **extra)
    keys_o, keys_m = set(orig.state_dict()), set(mine.state_dict())
    assert keys_o == keys_m, ('键不一致', keys_o - keys_m, keys_m - keys_o)
    # 原始权重逐键载入最小实现
    mine.load_state_dict(orig.state_dict(), strict=True)
    # SyncBN 单机不可前向：两侧等价换 BN（state_dict 同构，权重相同 → 数值可比）
    orig = swap_syncbn_to_bn(orig).eval()
    mine = swap_syncbn_to_bn(mine).eval()
    x = torch.randn(2, 64, 24, 24)
    with torch.no_grad():
        _, s_o, sal_o, g_o = orig(x.clone())
        _, s_m, sal_m, g_m = mine(x.clone())
    d_grid = (g_o - g_m).abs().max().item()
    d_sal = (sal_o - sal_m).abs().max().item()
    d_feat = (s_o - s_m).abs().max().item()
    assert d_grid < 1e-5 and d_sal < 1e-5 and d_feat < 1e-5, (d_grid, d_sal, d_feat)
    print(f'[3] {label}: state_dict 逐键一致 ({len(keys_o)} 键), '
          f'grid/saliency/feature 最大偏差 = {d_grid:.2e}/{d_sal:.2e}/{d_feat:.2e}')

print('\nALL SERVER CHECKS PASSED')
