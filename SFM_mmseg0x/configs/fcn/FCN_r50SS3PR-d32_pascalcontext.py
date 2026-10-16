dataset_type = 'PascalContextDataset59'
data_root = './data/VOCdevkit/VOC2010/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (513, 513)
img_scale = (2049, 513)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2049, 513), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(513, 513), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(513, 513), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 513),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
BATCH_SIZE = 16
GPU = 1
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=10,
    train=dict(
        type='PascalContextDataset59',
        data_root='./data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2049, 513), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(513, 513), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(513, 513), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='PascalContextDataset59',
        data_root='./data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2049, 513),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalContextDataset59',
        data_root='./data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2049, 513),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
ITER_SETTING = 40000
optimizer = dict(
    type='SGD',
    lr=0.004,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=2.0),
            SS=dict(lr_mult=2.0),
            pixelrelation=dict(lr_mult=2.0))))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
log_config = dict(
    interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
norm_cfg = dict(type='SyncBN', requires_grad=True)
LOG_DIR = None
SAMPLER_NUM = 3
model = dict(
    type='EncoderDecoderWithSaliencySamplerPixelRelation',
    log_dir=None,
    edge_loss_weight=100,
    use_adaptive_residual=False,
    pixel_relation=[1, 2, 4, 8, 16],
    sampler_num=3,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1cSaliencySampler',
        log_dir=None,
        sampler_num=3,
        sampler_strides=(1, 1, 1),
        sampler_paddings=(5, 5, 5),
        sampler_fwhm=(3, 3, 3),
        sampler_mode=('avgpsp_semantic_edge', 'avgpsp_semantic_edge',
                      'avgpsp_semantic_edge'),
        psp_ratio=4,
        use_lpf=False,
        align_corners=True,
        use_checkpoint=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=59,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=59,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = 'work_dirs'
gpu_ids = [0]
auto_resume = False
