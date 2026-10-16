# -*- coding: utf-8 -*-
# MMDetection 接入示例：Faster R-CNN R-50-FPN 只换 neck。
# 完整可运行配置 = 官方 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py + 以下改动。
# 适用于 mmdet 2.x / 3.x（mmseg 0.x / 1.x 同理，见 mmseg_fcn_r50_sfm.py）。
_base_ = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 导入即自动把 SFMFPN 注册进 mmdet 的 neck 注册表（mmdet 2.x → NECKS，3.x → MODELS）
custom_imports = dict(imports=['sfm'], allow_failed_imports=False)

model = dict(
    neck=dict(
        type='SFMFPN',           # 唯一必改项
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        sfm_compress_ratio=16),  # SFM 权重预测前的通道压缩比，越大越省算力
)
