import torch
import torch.nn as nn
# from ..segmentors.AdaConv import LocalPixelRelationConv
from mmseg_custom.models.segmentors.LPRM import LocalPixelRelationConv
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from mmseg_custom.models.segmentors.get_semantic_edge import GetSemanticEdge
# from mmseg.models.segmentors.bcr import BCR
# from mmseg.models.segmentors.guided_filter import ConvGuidedFilter
# from mmseg.models.segmentors.LIIF import LIIFDecoder
from torch.utils.tensorboard import SummaryWriter  
# import torch_dct as dct
# from mmseg.models.backbones.non_uniform_representation import NonUniformRepresentation
from mmcv.ops.carafe import CARAFEPack, xavier_init
# from mmseg.models.backbones.feature_align import DCNv2, CascadeImageGuideUpsample, CARAFEUpsample, CascadeImageFusion, torch_patchwise_dct, DCTSaver
# from mmseg_custom.models.backbones.interp2d import NonUniformInterpolation
from mmseg.models import EncoderDecoder
from mmseg.registry import MODELS
from torch import Tensor
from typing import List, Optional
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
import torch.nn.functional as F
import warnings

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class FocalAliasingLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fftshift(torch.fft.fft2(y, norm='ortho')) # freq[B, patch_n, c, h, w]
            # print(freq.shape)
            h, w = freq.size(-2), freq.size(-1)
            # freq[:, :, P, h//4:-h//4, w//4:-w//4] *= 0.
            freq[..., h//4:-h//4, w//4:-w//4] *= 0.
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
            raise NotImplementedError
        return freq

    def loss_formulation(self, recon_freq, real_freq=0, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2 ###
        # tmp = (recon_freq - real_freq).abs()
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, 0, matrix) * self.loss_weight
    
    def _forward(self, pred, matrix=None, **kwargs):
        aliasing_ratio = self.calculate_low_frequency_energy_ratio(pred)
        return aliasing_ratio.mean()
    
    def calculate_low_frequency_energy_ratio(self, tensor):
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor)
        # print(freq_domain_energy)
        # 计算低频成分的能量
        b, c, h, w = freq_domain_energy.shape
        low_freq_energy = freq_domain_energy[..., h//2 - h//4:h//2 + h//4, w//2 - w//4:w//2 + w//4].sum((-3, -2, -1)) 
        # print(low_freq_energy)
        # 计算低频成分能量与整个频域能量的比值
        total_energy = freq_domain_energy.sum((-3, -2, -1))  # 计算整个频域的能量
        low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
        # print(low_freq_energy_ratio)
        return 1.0 - low_freq_energy_ratio

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

# Copyright (c) OpenMMLab. All rights reserved.
def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


@MODELS.register_module()
class EncoderDecoderWithSaliencySamplerPixelRelation(EncoderDecoder):
    def __init__(self, 
                edge_loss_weight=100, edge_loss_weight_ratio=[1, 1, 1],
                log_dir=None,
                sampler_num=3, relation_kernel=3, pixel_relation=[1, 2, 4, 8, 16], 
                reverse_final=True,
                use_adaptive_residual=False, 
                aux_up_ratio=8, 
                pixel_relation_residual='', 
                use_pixel_shuffle=False,
                aliasing_loss_weight=0,
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.GetSemanticEdge = GetSemanticEdge()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        # if type(edge_loss_weight)
        self.edge_loss_weight = edge_loss_weight
        assert isinstance(edge_loss_weight_ratio, list)
        assert len(edge_loss_weight_ratio) == 3
        self.edge_loss_weight_ratio = edge_loss_weight_ratio

        self.reverse_final = reverse_final
        self.sampler_num = sampler_num
        self.aux_up_ratio = aux_up_ratio
        self.use_pixelrelation = (len(pixel_relation) > 0)
        self.pixel_relation_residual = pixel_relation_residual
        self.relation_kernel = relation_kernel
        # self.NUI = NonUniformInterpolation(kernel_size=3)
        if self.use_pixelrelation:
            self.pixelrelation_conv = nn.Sequential(
                nn.Conv2d(kargs['decode_head']['channels'], relation_kernel ** 2, kernel_size=3, stride=1, groups=1, dilation=1, padding=1),
                nn.SyncBatchNorm(relation_kernel ** 2),
                nn.ReLU(True)
            )

            pixelrelation = []
            for i in pixel_relation:
                pixelrelation.append(LocalPixelRelationConv(relation_kernel ** 2, kernel_size=relation_kernel, stride=1, groups=1, dilation=i, 
                                                            use_adaptive_residual=use_adaptive_residual))
            self.pixelrelation = nn.ModuleList(pixelrelation)

            # self.preds_weight = nn.parameter.Parameter(data=torch.Tensor([1 for _ in range(len(pixel_relation) + 1)]), requires_grad=True)
        self.use_pixel_shuffle = use_pixel_shuffle
        if use_pixel_shuffle:
            self.Temp = nn.parameter.Parameter(torch.Tensor([1.]))
            del self.decode_head.conv_seg
            self.decode_head.conv_seg = nn.Conv2d(self.decode_head.channels, self.decode_head.num_classes * 4, kernel_size=1)
            self.pixelshuffle = nn.PixelShuffle(2)

        self.aliasing_loss_weight = aliasing_loss_weight
        if aliasing_loss_weight > 0:
            self.AliasingLoss = FocalAliasingLoss(alpha=1.0)

    def get_aux_final_seg(self, x, grids, aux_head):
        """
        not used for mask2former
        """
        x = list(x)
        final_feat = aux_head._forward_feature(x)
        final_seg = aux_head.cls_seg(final_feat)
        final_grid = grids[-self.sampler_num].clone()
        if self.sampler_num > 2:
            for j in [1, 2, 3][-self.sampler_num + 1:-1]:
                # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners)
                final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)
            index = -1
        else:
            index = -1
        _, _, h, w = final_seg.shape
        final_grid = F.interpolate(final_grid, (h, w), mode='bilinear', align_corners=self.align_corners)
        if h % 2 == 1:
            final_seg = self.backbone.SS[index].reverse(final_seg, final_grid, 
                                                        h * self.aux_up_ratio - self.aux_up_ratio + 1, w * self.aux_up_ratio - self.aux_up_ratio + 1)
        else:
            final_seg = self.backbone.SS[index].reverse(final_seg, final_grid, h * self.aux_up_ratio, w * self.aux_up_ratio)
        return final_seg

    def get_final_feat(self, img):
        x, grids, smaps = self.extract_feat(img)
        x = list(x)
        _outs = self.decode_head._forward_feature(x)
        if isinstance(_outs, tuple):
            final_feat, cls_list = _outs # PCAA
        else:
            final_feat = _outs
            cls_list = None

        final_seg = self.decode_head.cls_seg(final_feat)
        if self.use_pixel_shuffle:final_seg *= self.Temp
        if self.use_pixelrelation:
            pixelrelation_feat = self.pixelrelation_conv(final_feat)
            final_feat = torch.cat([final_seg, pixelrelation_feat], dim=1)
        else:
            final_feat = final_seg

        final_grid = grids[-self.sampler_num].clone()
        if self.reverse_final:
            # print(self.align_corners)
            if self.sampler_num > 1:
                for j in [1, 2, 3][-self.sampler_num + 1:]:
                    # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                    # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners)
                    final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)

            b, _, h, w =x[0].shape
            # up_scale = (w - 1) // (final_feat.size(-1) - 1)
            final_feat = F.interpolate(final_feat, final_grid.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # final_feat = F.interpolate(final_feat, x[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # _, _, h, w = final_feat.shape
            # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 4 - 3 * (h % 2), w * 4 - 3 * (w % 2))
            # print(final_feat.shape, x[0].shape)
            final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h, w)
            # up_scale = int(round(w / final_feat.size(-1), ndigits=1))
            # final_feat = F.interpolate(final_feat, x[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # final_grid = F.interpolate(final_grid, x[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # final_feat = F.interpolate(final_feat, x[0].shape[-2:], mode='nearest')
            # final_grid = F.interpolate(final_grid, x[0].shape[-2:], mode='nearest')
            # uniform_grid = self.backbone.SS[-3].gen_coord_grid_without_pad(h, w, device=final_feat) * 2 - 1.0
            # uniform_grid = uniform_grid.expand(b, -1, -1, -1)
            # final_feat = self.NUI(final_feat, non_uniform_grid=final_grid, uniform_grid=uniform_grid, up_scale=up_scale)
        
        mid_segs = []
        if self.use_pixelrelation:
            # pixelrelation_feat = final_feat[:, -self.relation_kernel ** 2:, ]
            # final_seg = final_feat[:, :-self.relation_kernel ** 2, ]
            final_seg = final_feat[:, :self.num_classes, ]
            pixelrelation_feat = final_feat[:, self.num_classes:, ]

            # mid_seg = final_seg
            # print(self.preds_weight) # [0.8167, 0.6941, 0.4476, 0.4421, 1.3931, 0.8089]
            # preds_weight = self.preds_weight.softmax(-1)
            # print(preds_weight) #[0.1661, 0.1448, 0.1138, 0.1162, 0.2919, 0.1671])
            # final_seg = preds_weight[-1] * final_seg
            # reidual_final_seg = final_seg
            mid_segs.append(final_seg)
            for i in range(len(self.pixelrelation)):
                # mid_segs.append(final_seg)
                if self.pixel_relation_residual == '':
                    pixelrelation_feat, final_seg = self.pixelrelation[i](pixelrelation_feat, final_seg, use_checkpoint=True)
                elif self.pixel_relation_residual == 'avg_weight':
                    # pixelrelation_feat, mid_seg = self.pixelrelation[i](pixelrelation_feat, mid_seg, use_checkpoint=True)
                # pixelrelation_feat = ((i+1) * pixelrelation_feat + F.relu(_pixelrelation_feat)) / (i+1 + 1)
                    pixelrelation_feat, _final_seg = self.pixelrelation[i](pixelrelation_feat, final_seg, use_checkpoint=True)
                    final_seg = ((i+1) * final_seg + _final_seg) / (i+1 + 1)
                    # print(final_seg)
                elif self.pixel_relation_residual == 'feat_residual':
                # pixelrelation_feat, mid_seg = self.pixelrelation[i](pixelrelation_feat, mid_seg, use_checkpoint=True)
                # pixelrelation_feat = ((i+1) * pixelrelation_feat + F.relu(_pixelrelation_feat)) / (i+1 + 1)
                    _pixelrelation_feat, final_seg = self.pixelrelation[i](pixelrelation_feat, final_seg, use_checkpoint=True)
                    # final_seg = ((i+1) * final_seg + _final_seg) / (i+1 + 1)
                    pixelrelation_feat = F.relu(pixelrelation_feat + _pixelrelation_feat)
                    # print(final_seg)
                else:
                    raise NotImplementedError
                # final_seg = final_seg + mid_seg * preds_weight[i]
                # final_seg = 0.5 * final_seg
            # final_seg = final_seg / (len(self.pixelrelation) + 1)
            # final_seg = reidual_final_seg + final_seg
        else:
            final_seg = final_feat

        # pixelrelation_feat, final_seg = self.pixelrelation[4](pixelrelation_feat, final_seg, use_checkpoint=True)
        # """
        if self.use_pixel_shuffle:final_seg = self.pixelshuffle(final_seg)
        return x, grids, smaps, final_seg, final_grid, cls_list, mid_segs

    def _encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, grids, smaps, final_feat, _, _, _ = self.get_final_feat(img) # use only final feat for inference
        # out = self.decode_head.cls_seg(final_feat)
        # print(final_feat.shape)
        out = final_feat

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    # def forward_train(self, img, img_metas, gt_semantic_seg):
    def _loss(self, img, data_samples):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # print(data_samples)
        # print(data_samples[0].gt_sem_seg.data)
        gt_semantic_seg = torch.stack([i.gt_sem_seg.data for i in data_samples], dim=0)
        # print(gt_semantic_seg.shape)

        losses = dict()
        x, grids, smaps, final_feat, final_grid, cls_list, mid_segs = self.get_final_feat(img)
        seg_logits = final_feat
        # seg_logits = self.decode_head.cls_seg(final_feat)
        if cls_list is not None: # for PCAA
            deformed_seg_label = gt_semantic_seg.clone()
            final_grid = F.interpolate(final_grid, deformed_seg_label.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            deformed_seg_label = F.grid_sample(deformed_seg_label.float(), final_grid.permute(0, 2, 3, 1).detach(), mode='nearest').long()
            loss_decode = self.decode_head.losses(seg_logits, cls_list, gt_semantic_seg, deformed_seg_label)
            # loss_decode = self.decode_head.losses(seg_logits, cls_list, gt_semantic_seg, gt_semantic_seg)
        else:
            loss_decode = self._decode_head_forward_train(x, data_samples)
            # loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        # print(loss_decode)
        losses.update(loss_decode)

        # for i, mid_seg in enumerate(mid_segs):
        # #     # aux head 0.4
        #     loss_mid = self.decode_head.losses(mid_seg, gt_semantic_seg)
        #     for k in loss_mid:
        #         if 'loss' in k:loss_mid[k] *= 0.4
        #     losses.update(add_prefix(loss_mid, f'mid_{i}'))
        """
        if coarse_seg != None:
            deformed_gt_semantic_seg = gt_semantic_seg.clone()
            for j in [1, 2, 3][-self.sampler_num:]:
                grid_temp = F.interpolate(grids[j].clone(), gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=True)
                deformed_gt_semantic_seg = F.grid_sample(deformed_gt_semantic_seg.float(), grid_temp.detach().permute(0, 2, 3, 1), mode='nearest', align_corners=True).long()
                coarse_decode = self.decode_head.losses(coarse_seg, deformed_gt_semantic_seg)
                losses.update(add_prefix(coarse_decode, 'coarse'))
        """
        """
        gt_segs = [gt_semantic_seg.clone() ]
        for j in [1, 2, 3][-self.sampler_num:]:
            grid_temp = F.interpolate(grids[j].clone(), gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=True)
            gt_semantic_seg = F.grid_sample(gt_semantic_seg.float(), grid_temp.detach().permute(0, 2, 3, 1), mode='nearest', align_corners=True).long()
            gt_segs.append(gt_semantic_seg.clone())

        gt_segs = gt_segs[::-1]
        # print(len(gt_segs), len(refine_segs))
        # need to reverse
        for idx, (refine_seg, gt_seg) in enumerate(zip(refine_segs, gt_segs)):
            loss_decode = self.decode_head.losses(refine_seg, gt_seg)
            _, _, _h, _w = refine_seg.shape
            for k in loss_decode: 
                if "loss" in k:
                    if idx == (len(refine_segs) - 1):
                        losses[f'{k}_{_h}'] = loss_decode[k] * 1.0
                    else:
                        losses[f'{k}_{_h}'] = loss_decode[k] * 0.2
                else:
                    losses[f'{k}_{_h}'] = loss_decode[k]
            losses.update(loss_decode)
        #"""
        
        # auxiliary_head index = 2
        # if self.with_auxiliary_head:
        #     deformed_gt_semantic_seg = gt_semantic_seg.clone()
        #     for j in [1, 2, 3][-self.sampler_num:-1]:
        #         grid_temp = F.interpolate(grids[j].clone(), gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=True)
        #         deformed_gt_semantic_seg = F.grid_sample(deformed_gt_semantic_seg.float(), grid_temp.detach().permute(0, 2, 3, 1), mode='nearest', align_corners=True).long()
        #     loss_aux = self._auxiliary_head_forward_train(
        #         x, img_metas, deformed_gt_semantic_seg)
        #     losses.update(loss_aux)

        if self.with_auxiliary_head:
            aux_seg = self.get_aux_final_seg(x, grids, self.auxiliary_head)
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            # loss_aux = self.auxiliary_head.losses(aux_seg, gt_semantic_seg)
            losses.update(add_prefix(loss_aux, f'aux'))

        semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
        # b, c, h, w = semantic_edge.shape
        # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
        semantic_edge[semantic_edge > 0] = 1.
        semantic_edge[semantic_edge <= 0] = 0.001


        for j in [1, 2, 3][-self.sampler_num:]:
            b, c, h, w = grids[j].shape
            if j == [1, 2, 3][-self.sampler_num]:
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
                semantic_edge = torch.nn.AdaptiveAvgPool2d(grids[j].shape[-2:])(semantic_edge)
            else:
                semantic_edge = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(semantic_edge)
                semantic_edge = F.interpolate(semantic_edge, grids[j].shape[-2:], mode='bilinear', align_corners=self.align_corners)
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            # semantic_edge_softmax = semantic_edge.clone().view(-1, h * w).softmax(-1).view(b, 1, h, w)
            # print(semantic_edge_softmax.max(), semantic_edge_softmax.min())
            # semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge_softmax, h, w)
            semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge, h, w).detach()
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            losses[f'semantic_edge_loss_{j}'] = self.edge_loss_weight * self.edge_loss_weight_ratio[j - 1] * torch.clamp(
                F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
            ).mean()
            # losses[f'semantic_edge_loss_{j}'] = self.edge_loss_weight * F.l1_loss(grids[j], semantic_edge_grid)        
            # losses[f'semantic_edge_loss_{j}'] = 1 * F.mse_loss(grids[j], semantic_edge_grid)        
            if self.log_dir is not None:
                # print(gt_semantic_seg.shape, gt_semantic_seg)
                # print(semantic_edge.shape, semantic_edge.mean(dim=1))
                # semantic_edge_vis = F.interpolate(semantic_edge, , mode='bilinear', align_corners=True)
                # semantic_edge_grid_vis = F.interpolate(semantic_edge_grid, (semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16) , mode='bilinear', align_corners=True)
                semantic_edge_grid_vis = semantic_edge_grid
                semantic_edge_grid_vis = (semantic_edge_grid_vis - semantic_edge_grid_vis.min()) / (semantic_edge_grid_vis.max() - semantic_edge_grid_vis.min())
                self.writer.add_images(f"sedge_gridx{h}x{w}", semantic_edge_grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
                self.writer.add_images(f"sedge_gridy{h}x{w}", semantic_edge_grid_vis[:, 1:], global_step=None, walltime=None, dataformats='NCHW')

                # semantic_edge_vis = torch.nn.AdaptiveMaxPool2d((semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16))(semantic_edge)
                semantic_edge_vis = semantic_edge
                self.writer.add_images(f"semantic_edge{h}x{w}", (semantic_edge_vis - semantic_edge_vis.min()) / (semantic_edge_vis.max() - semantic_edge_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
                pass
            semantic_edge = F.grid_sample(semantic_edge, grids[j].detach().permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners) 
        # final_feat = self.backbone.SS[j]
        for k in losses:
            if 'loss' in k:
                # losses[k] = torch.clamp(losses[k], 0, 1.)
                if torch.isnan(losses[k]):
                    # losses[k] = losses['semantic_edge_loss_3']
                    raise ValueError(f"Loss {k} is NAN!!!")
                    # losses[k] = 0.
                    # del losses[k]
        # if self.use_awl:losses = self.AWL(losses)
        return losses
    # do not interfere ori extract_feat
    def _extract_feat(self, img):
        """Extract features from images."""
        x, grids, smaps = self.backbone(img)

        if self.log_dir is not None:
            img_vis = F.interpolate(img, (img.size(-2) // 8, img.size(-1) // 8), mode='bilinear', align_corners=True)
            self.writer.add_image("img", (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
            pass

        if self.with_neck:
            # x = self.neck(x)
            raise NotImplementedError
        return x, grids, smaps

    def extract_feat(self, img):
        """Extract features from images."""
        x, grids, smaps = self.backbone(img)
        # print(type(smaps))
        if self.with_neck:
            x = self.neck(x, grids)
        return x, grids, smaps

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, grids, smaps = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict([x, grids, smaps], batch_img_metas,
                                              self.test_cfg)

        return seg_logits
    
    # 1
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x, grids, smaps = self._extract_feat(inputs)
        return self.decode_head.forward(x)
    # 2
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples) 
    # 3
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x, grids, smaps = self.extract_feat(inputs)

        losses = dict()

        if self.aliasing_loss_weight > 0:
            x_sampleds = smaps['x_sampleds']
            x_sampleds = [i for i in x_sampleds if i != None]
            aliasing_losses = {f'aliasing_loss_{i}':self.AliasingLoss(feat) * self.aliasing_loss_weight for i, feat in enumerate(x_sampleds)} 
            # aliasing_losses = {f'aliasing_loss_{i}':self.AliasingLoss(feat.mean(dim=1, keepdim=True)) for i, feat in enumerate(x_sampleds)}
            # print(aliasing_losses)
            losses.update(aliasing_losses)

        loss_decode = self._decode_head_forward_train([x, grids, smaps], data_samples) ###
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        gt_semantic_seg = torch.stack([i.gt_sem_seg.data for i in data_samples], dim=0)
        semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
        # b, c, h, w = semantic_edge.shape
        # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
        semantic_edge[semantic_edge > 0] = 1.
        semantic_edge[semantic_edge <= 0] = 0.001

        not_none_list = []
        for idx in range(len(grids)):
            if grids[idx] is not None: 
                not_none_list.append(idx)
        not_none_list = [1, 2, 3]
        for j in not_none_list[-self.sampler_num:]:
            b, c, h, w = grids[j].shape
            if j == not_none_list[-self.sampler_num]:
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
                semantic_edge = torch.nn.AdaptiveAvgPool2d(grids[j].shape[-2:])(semantic_edge)
            else:
                semantic_edge = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(semantic_edge)
                semantic_edge = F.interpolate(semantic_edge, grids[j].shape[-2:], mode='bilinear', align_corners=self.align_corners)
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            # semantic_edge_softmax = semantic_edge.clone().view(-1, h * w).softmax(-1).view(b, 1, h, w)
            # print(semantic_edge_softmax.max(), semantic_edge_softmax.min())
            # semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge_softmax, h, w)
            semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge, h, w).detach()
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            losses[f'semantic_edge_loss_{j}'] = self.edge_loss_weight * self.edge_loss_weight_ratio[j - 1] * torch.clamp(
                F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
            ).mean()
            # losses[f'semantic_edge_loss_{j}'] = self.edge_loss_weight * F.l1_loss(grids[j], semantic_edge_grid)        
            # losses[f'semantic_edge_loss_{j}'] = 1 * F.mse_loss(grids[j], semantic_edge_grid)        
            if self.log_dir is not None:
                # print(gt_semantic_seg.shape, gt_semantic_seg)
                # print(semantic_edge.shape, semantic_edge.mean(dim=1))
                # semantic_edge_vis = F.interpolate(semantic_edge, , mode='bilinear', align_corners=True)
                # semantic_edge_grid_vis = F.interpolate(semantic_edge_grid, (semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16) , mode='bilinear', align_corners=True)
                semantic_edge_grid_vis = semantic_edge_grid
                semantic_edge_grid_vis = (semantic_edge_grid_vis - semantic_edge_grid_vis.min()) / (semantic_edge_grid_vis.max() - semantic_edge_grid_vis.min())
                self.writer.add_images(f"sedge_gridx{h}x{w}", semantic_edge_grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
                self.writer.add_images(f"sedge_gridy{h}x{w}", semantic_edge_grid_vis[:, 1:], global_step=None, walltime=None, dataformats='NCHW')

                # semantic_edge_vis = torch.nn.AdaptiveMaxPool2d((semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16))(semantic_edge)
                semantic_edge_vis = semantic_edge
                self.writer.add_images(f"semantic_edge{h}x{w}", (semantic_edge_vis - semantic_edge_vis.min()) / (semantic_edge_vis.max() - semantic_edge_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
                pass
            semantic_edge = F.grid_sample(semantic_edge, grids[j].detach().permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners) 
        return losses