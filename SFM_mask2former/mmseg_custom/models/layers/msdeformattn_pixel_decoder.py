# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from torch import Tensor

from mmseg.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
# from ..task_modules.prior_generators import MlvlPointGenerator
# from .positional_encoding import SinePositionalEncoding
# from .transformer import Mask2FormerTransformerEncoder
from mmdet.models.layers.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from mmseg_custom.models.backbones.SFM_fpn import AlignModule, LPRMAlignUpModule, compute_real_grids

@MODELS.register_module()
class MSDeformAttnPixelDecoderSFM(MSDeformAttnPixelDecoder):
    """
    集成了 SFM 模块的 MSDeformAttnPixelDecoder，参考 NUAS 结构实现。
    """
    def __init__(self,
                # SFM-specific parameters
                 align_corners: bool = False,
                 use_align: bool = True,
                 compress_ratio: int = 16,
                 use_last_align=True,
                 post_align=False,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        feat_channels = kwargs.get('feat_channels')
        in_channels_list = kwargs.get('in_channels')
        self.align_corners = align_corners
        self.use_align = use_align
        self.use_last_align = use_last_align
        self.post_align = post_align
        self.AS_aligns = nn.ModuleList()
        self.self_aligns = nn.ModuleList()
        for i in range(self.num_input_levels):
            self_align = AlignModule(feat_channels, k=3, align_corners=align_corners, compress_ratio=compress_ratio)
            self.self_aligns.append(self_align)
        if self.use_align:
            for i in range(2):
                align_module = LPRMAlignUpModule(feat_channels, feat_channels, align_corners=align_corners, compress_ratio=compress_ratio)
                self.AS_aligns.append(align_module)
        if self.use_last_align:
            self.AS_align_last = LPRMAlignUpModule(feat_channels, feat_channels, align_corners=align_corners, compress_ratio=compress_ratio)

    def forward(self, feats: List[Tensor], grids, smaps) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - multi_scale_features (list[Tensor]): Multi scale \
                        features, each in shape (batch_size, c, h, w).
        """
        real_grids = compute_real_grids(grids, feats[0].shape, self.align_corners)

        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        
        ##### 
        feat_projected_list = []
        # print('self.num_encoder_levels', self.num_encoder_levels)
        # self.num_encoder_levels: 3
        # level_idx: 3, 2, 1
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            feat_projected = self.self_aligns[level_idx](feat_projected, real_grids[level_idx])
            feat_projected_list.append(feat_projected)
        feat_projected_list = feat_projected_list[::-1]

        if self.use_align and not self.post_align:
            for i in range(self.num_encoder_levels - 1):
                level_idx = len(feat_projected_list) - i - 1
                # feat = feats[level_idx]
                # feat_projected_list[level_idx - 1]
                # feat_projected_list[level_idx]
                # self.AS_aligns[i]
                x_low_upsampled = self.AS_aligns[i](x_low=feat_projected_list[level_idx], guidance_high_aligned=feat_projected_list[level_idx - 1])
                feat_projected_list[level_idx - 1] = x_low_upsampled + feat_projected_list[level_idx - 1]
                # h, w = feat.shape[-2:]
                # feat_projected_list.append(feat_projected)

        # level_idx: 3, 2, 1
        # feat_projected_list: [1, 2, 3]
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            # feat_projected = self.input_convs[i](feat)
            feat_projected = feat_projected_list[level_idx - 1]
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized) ###
            # pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_queries),
        # total_num_queries=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_queries, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=1)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_queries, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_radios)
        # (batch_size, c, num_total_queries)
        memory = memory.permute(0, 2, 1)

        # from low resolution to high resolution
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        # if self.use_align and self.post_align:
        #     # outs: 2, 1, 0
        #     for i in range(self.num_encoder_levels - 1):
        #         level_idx = i
        #         # feat = feats[level_idx]
        #         # feat_projected_list[level_idx - 1]
        #         # feat_projected_list[level_idx]
        #         # self.AS_aligns[i]
        #         # print(outs[level_idx].shape)
        #         # print(outs[level_idx + 1].shape)
        #         hr_x, lr_x = self.AS_aligns[i](hr_x=outs[level_idx + 1].contiguous(), lr_x=outs[level_idx].contiguous(), use_checkpoint=False)
        #         outs[level_idx + 1] = hr_x + lr_x
        #         # h, w = feat.shape[-2:]
        #         # feat_projected_list.append(feat_projected)

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            cur_feat = self.self_aligns[i](cur_feat, real_grids[i])
            if self.use_last_align:
                x_low_upsampled = self.AS_align_last(x_low=outs[-1].contiguous(), guidance_high_aligned=cur_feat)
                # x_low_upsampled = self.AS_align_last(hr_x=cur_feat, lr_x=outs[-1].contiguous(), use_checkpoint=False)
                y = cur_feat + x_low_upsampled
            else:
                y = cur_feat + F.interpolate(
                    outs[-1],
                    size=cur_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[:self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features
