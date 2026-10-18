_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

train_dataloader = dict(
    batch_size=8, 
    num_workers=8,
)
custom_imports = dict(imports=['mmdet.models', 
                               'mmseg_custom.models.backbones.swin_as',
                               'mmseg_custom.models.layers.msdeformattn_pixel_decoder',
                               'mmseg_custom.models.decode_heads.mask2former_head_custom',
                               'mmseg_custom.models.segmentors.encoder_decoder_custom',
                               ], allow_failed_imports=False)
LOG_DIR = None
SAMPLER_NUM = 3
ALIGN_CORNERS = False
depths = [2, 2, 6, 2]
model = dict(
    # type='EncoderDecoder',
    type='EncoderDecoderWithSaliencySamplerPixelRelation',
    log_dir=LOG_DIR, 
    edge_loss_weight=100,
    # pixel_relation=[1, 2, 4, 8, 16, 32, 64],
    pixel_relation=[],
    sampler_num=SAMPLER_NUM,
    backbone=dict(
        _delete_=True,
        # type='SwinTransformer',
        type='SwinTransformerSaliencySampler',
        use_checkpoint=False,
        align_corners=ALIGN_CORNERS,
        log_dir=LOG_DIR, 
        sampler_num =SAMPLER_NUM, 
        sampler_strides=(1, 1, 1), 
        sampler_paddings=(5, 5, 5), 
        sampler_fwhm=(3, 3, 3), 
        sampler_mode=('avg_lhpf_psp', 'avg_lhpf_psp', 'avg_lhpf_psp'),
        psp_ratio=16,
        with_cp=False,
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        # type='Mask2FormerHeadNonUniformEmbedding',
        type='Mask2FormerHeadNonUniformSFM',
        in_channels=[96, 192, 384, 768],
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoderSFM',
            compress_ratio=16, ###
            align_corners = False,
            use_align = True,
            use_last_align=True,
        ),
    )
)

backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,

    'backbone.SS': dict(lr_mult=1.0, decay_mult=1.0), ###
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))