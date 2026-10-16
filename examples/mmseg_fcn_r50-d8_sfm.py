# -*- coding: utf-8 -*-
# MMSegmentation 接入示例：FCN R-50 只换 neck（mmseg 0.x / 1.x 通用）。
# 完整可运行配置 = 官方 configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py + 以下改动。
_base_ = 'fcn/fcn_r50-d8_512x1024_40k_cityscapes.py'

# 导入即自动把 SFMFPN 注册进 mmseg 的 neck 注册表（0.x → NECKS，1.x → MODELS）
custom_imports = dict(imports=['sfm'], allow_failed_imports=False)

model = dict(
    neck=dict(
        type='SFMFPN',           # 唯一必改项
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,
        num_outs=1,
        sfm_compress_ratio=16),
)
