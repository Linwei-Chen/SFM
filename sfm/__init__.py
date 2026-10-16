# -*- coding: utf-8 -*-
"""sfm —— Spatial Frequency Modulation 即插即用算子包（TPAMI 2025）。

安装::

    pip install .          # 或 pip install git+https://github.com/Linwei-Chen/SFM.git

导入即用（纯 PyTorch，无框架也可用）::

    from sfm import ARS, LPRMAlignUpModule, AlignModule, SFMFPN

接入 MMDetection / MMSegmentation（导入时自动注册 SFMFPN 到框架 neck 注册表）::

    # config 里加：
    custom_imports = dict(imports=['sfm'], allow_failed_imports=False)
    # neck 改为：
    neck=dict(type='SFMFPN', in_channels=[...], out_channels=256, num_outs=5)

接入点（对应论文两条发布路线）：
  1. backbone 下采样（接入点 A）: ``sfm.ARS``（自适应重采样 SaliencySampler）
  2. FPN 上采样（接入点 B）      : ``sfm.SFMFPN`` / ``sfm.LPRMAlignUpModule``
     （ARS + CARAFE 式重组 + 三点动态采样加权，论文 FPN 线）
  3. 非均匀网格摊平              : ``sfm.AlignModule`` + ``sfm.compute_real_grids``
  4. FCN style 非均匀上采样（MSAU，论文 FCN 线）::
       sfm.SaliencySampler.reverse —— qhull 三点重心插值（见 :mod:`sfm.interpolate`）

参考论文:
    Spatial Frequency Modulation for Semantic Segmentation (TPAMI 2025)
    https://arxiv.org/abs/2507.11893
"""

from .upsampling import (
    LPRMAlignUpModule,
    AlignModule,
    LocalPixelRelationConv,
    LPRM_efficient,
    carafe_reassembly,
    swap_syncbn_to_bn,
)
from .adaptive_sampling import (
    ARS,
    SaliencySampler,
    makeGaussian,
    PSP,
    PSP_Module,
    LHPFConv3,
    compute_real_grids,
)
from .fpn import SFMFPN, register_sfm, FPN_BASE_SOURCE
from .interpolate import (
    Interp2D,
    reverse_feat,
    reverse_with_inv_grid,
    reverse_grid,
    invert_nonseparable_grid,
    qhull_backend,
)

__version__ = '1.1.0'

__all__ = [
    # 上采样/对齐（FPN 接入点 B）
    'LPRMAlignUpModule', 'AlignModule', 'LocalPixelRelationConv',
    'LPRM_efficient', 'carafe_reassembly', 'swap_syncbn_to_bn',
    # 自适应采样（backbone 接入点 A）
    'ARS', 'SaliencySampler', 'makeGaussian', 'PSP', 'PSP_Module',
    'LHPFConv3', 'compute_real_grids',
    # FCN style 非均匀上采样（qhull MSAU，论文 FCN 线）
    'Interp2D', 'reverse_feat', 'reverse_with_inv_grid', 'reverse_grid',
    'invert_nonseparable_grid', 'qhull_backend',
    # FPN
    'SFMFPN', 'register_sfm', 'FPN_BASE_SOURCE',
    '__version__',
]
