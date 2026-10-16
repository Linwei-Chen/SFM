# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize

from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .img_preprocess import *
# from ..segmentors.AdaConv import LocalPixelRelationConv
from mmseg.models.segmentors.AdaConv import LocalPixelRelationConv
from mmcv.cnn import ConvModule
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from mmseg.models.segmentors.get_semantic_edge import GetSemanticEdge
from mmseg.models.segmentors.bcr import BCR
# from mmseg.models.segmentors.guided_filter import ConvGuidedFilter
from mmseg.models.segmentors.LIIF import LIIFDecoder
from torch.utils.tensorboard import SummaryWriter  
# import torch_dct as dct
from mmseg.models.backbones.non_uniform_representation import NonUniformRepresentation
from mmcv.ops.carafe import CARAFEPack, xavier_init
from mmseg.models.backbones.feature_align import DCNv2, CascadeImageGuideUpsample, CARAFEUpsample, CascadeImageFusion, torch_patchwise_dct, DCTSaver
from mmseg.models.backbones.interp2d import NonUniformInterpolation
from mmseg.models.losses.focal_fft_loss import FocalAliasingLoss
class FeatSaver:
    def __init__(self, save_path='/home/ubuntu/code/ResolutionDet/mmseg_exp/utils/sfpn_dct.npy'):
        self.dct_list = []
        self.save_path = save_path

    def save(self, x, print_avg=True, log=False):
        self.dct_list.append(x)
        all = torch.stack(self.dct_list, dim=0)
        # if log:
        #     avg = avg.log()
        #     avg[avg < 0] = 0
        # avg = avg.numpy()
        if print_avg:print(self.save_path, len(self.dct_list))
        # np.save(self.save_path, avg)
        np.save(self.save_path, all.numpy)

    def get_avg(self):
        avg = torch.stack(self.dct_list, dim=0).mean(dim=0)
        return avg
        
@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 use_awl=False,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.use_awl = use_awl
        if self.use_awl:
            self.AWL = AutomaticWeightedLoss(10)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        if self.use_awl:losses = self.AWL(losses)
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=10):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def _forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

    def forward(self, losses):
        for i, key in enumerate(losses.keys()):
            if 'dct' in key or 'freq' in key:
                losses[key] = 0.5 / (self.params[i] ** 2) * losses[key] + torch.log(1 + self.params[i] ** 2)
            else:
                losses[key] = 1.0 / (self.params[i] ** 2) * losses[key] + torch.log(1 + self.params[i] ** 2)
        return losses

@SEGMENTORS.register_module()
class EncoderDecoderWithDCT(EncoderDecoder):
    def __init__(self, patch_size=8, compress_channel=0, dropout_ratio=-1, use_lpr=False, use_idct=True, *args, **kargs):
        super().__init__(*args, **kargs)
        self.patch_size = patch_size
        if compress_channel <= 0:
            self.comp_conv = nn.Identity()
            self.conv_dct = nn.Conv2d(self.decode_head.channels, self.num_classes * self.patch_size ** 2, kernel_size=1)
        else:
            self.comp_conv = nn.Sequential(
                nn.Conv2d(self.decode_head.channels, compress_channel, kernel_size=1),
                # nn.SyncBatchNorm(compress_channel),
                # nn.ReLU(True),
                # nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
            )
            self.conv_dct = nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1)
        self.Temp = nn.parameter.Parameter(torch.Tensor([1.]))
        self.use_lpr = use_lpr
        self.use_idct = use_idct
        if use_lpr:
            self.LPR = LocalPixelRelationConv(self.num_classes, kernel_size=3, stride=1, groups=1, dilation=patch_size)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        # self.AWL = AutomaticWeightedLoss(10)
    # def cls_seg(self, feat):
    #     """Classify each pixel."""
    #     if self.dropout is not None:
    #         feat = self.dropout(feat)
    #     output = self.conv_seg(feat)
    #     return output

    def dct_seg(self, img, head_feat):
        """Forward function."""
        if self.dropout is not None:
            head_feat = self.dropout(head_feat)
        output = self.conv_dct(self.comp_conv(head_feat))
        output = nn.PixelShuffle(self.patch_size)(output)
        b, c, h, w = output.shape
        # print(output.shape)
        output = torch.nn.functional.unfold(output, (self.patch_size , self.patch_size), stride=self.patch_size) # c, c * 8 * 8, h // 8, w // 8
        output = output.reshape(b, self.num_classes, self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        output = output.permute(0, 1, 3, 4, 2)
        output = output.reshape(b, -1, self.patch_size, self.patch_size)
        # print(output.shape)
        # exit()
        # output = output.reshape(b, -1, 8, 8)
        if self.use_idct:
            output = dct.idct_2d(output) * 8
        output = output * self.Temp
        output = output.reshape(b, self.num_classes, h // self.patch_size, w // self.patch_size, self.patch_size * self.patch_size)
        output = output.permute(0, 1, 4, 2, 3)
        output = output.reshape(b, self.num_classes * self.patch_size * self.patch_size, h // self.patch_size * w // self.patch_size)
        output = torch.nn.functional.fold(output, (h, w), (self.patch_size, self.patch_size), stride=self.patch_size)
        return output


    def get_final_feats(self, img):
        x = self.extract_feat(img)
        # out = self._decode_head_forward_test(x, img_metas)
        head_feat=self.decode_head._forward_feature(x)
        return x, head_feat

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)
        # # out = self._decode_head_forward_test(x, img_metas)
        # head_feat=self.decode_head._forward_feature(x)
        x, head_feat = self.get_final_feats(img)
        initial_seg = self.decode_head.cls_seg(head_feat)
        dct_seg = self.dct_seg(img, head_feat)
        # img_dct = 
        initial_seg = resize(
            input=initial_seg,
            # size=img.shape[2:],
            size=dct_seg.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        out = initial_seg + dct_seg
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        # x = self.extract_feat(img)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        # seg_logits = self.encode_decode(img, img_metas)
        x, head_feat = self.get_final_feats(img)
        initial_seg = self.decode_head.cls_seg(head_feat)
        dct_seg = self.dct_seg(img, head_feat)
        initial_seg = resize(
            input=initial_seg,
            size=dct_seg.shape[2:],
            # size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits = initial_seg + dct_seg

        losses = dict()
        loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # print(losses)
        # losses = self.AWL(losses)
        # print(self.AWL.params)
        if self.use_awl:losses = self.AWL(losses)
        return losses

def distance(i, j, rows, cols, r):
    rows = rows - 1
    cols = cols - 1
    # rows = rows - 0.5 if rows % 2 == 0 else rows
    # cols = cols - 0.5 if cols % 2 == 0 else cols
    dis = np.sqrt((i - rows/2) ** 2 + (j - cols/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, rows, cols, r=r)
    return mask

@SEGMENTORS.register_module()
class EncoderDecoderFreqRemove(EncoderDecoder):
    def __init__(self, r=64, mode='final', *args, **kargs):
        super().__init__(*args, **kargs)
        self.r = r
        self.mode = mode
        self.mse_list = []
        self.cos_list = []

    def get_final_feats(self, img):
        x = self.extract_feat(img)
        head_feat=self.decode_head._forward_feature(x)
        # head_feat=self.auxiliary_head._forward_feature(x)
        
        head_feat = resize(input=head_feat, scale_factor=2, mode='nearest').float()
        # head_feat = resize(input=head_feat,scale_factor=2, mode='bilinear')
        low_part = torch.fft.fftshift(torch.fft.fft2(head_feat))
        mask = mask_radial(np.zeros([low_part.size(-2), low_part.size(-1)]), r=self.r)
        mask = torch.Tensor(mask)[None, None].to(head_feat.device)
        head_feat = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
        seg = self.decode_head.conv_seg(head_feat)
        # seg = self.auxiliary_head.conv_seg(head_feat)
        return x, head_feat, seg, None

    def get_aux_final_feats(self, img):
        x = self.extract_feat(img)
        # head_feat=self.decode_head._forward_feature(x)
        head_feat=self.auxiliary_head._forward_feature(x).float()
        
        # head_feat = resize(input=head_feat,scale_factor=2, mode='nearest')
        # head_feat = resize(input=head_feat,scale_factor=2, mode='bilinear')
        low_part = torch.fft.fftshift(torch.fft.fft2(head_feat))
        mask = mask_radial(np.zeros([low_part.size(-2), low_part.size(-1)]), r=self.r)
        mask = torch.Tensor(mask)[None, None].to(head_feat.device)
        head_feat = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
        # seg = self.decode_head.conv_seg(head_feat)
        seg = self.auxiliary_head.conv_seg(head_feat)
        return x, head_feat, seg, None

    def get_fused_final_feats(self, img):
        x = self.extract_feat(img)
        head_feat=self.decode_head._forward_feature(x)
        head_feat = resize(input=head_feat,scale_factor=2, mode='nearest').float()
        # head_feat = resize(input=head_feat,scale_factor=2, mode='bilinear')
        low_part = torch.fft.fftshift(torch.fft.fft2(head_feat))
        mask = mask_radial(np.zeros([low_part.size(-2), low_part.size(-1)]), r=self.r)
        mask = torch.Tensor(mask)[None, None].to(head_feat.device)
        head_feat_low = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
        seg = self.decode_head.conv_seg(head_feat_low)
        # seg = self.auxiliary_head.conv_seg(head_feat)

        aux_head_feat=self.auxiliary_head._forward_feature(x).float()
        aux_low_part = torch.fft.fftshift(torch.fft.fft2(aux_head_feat))
        mask = mask_radial(np.zeros([aux_low_part.size(-2), aux_low_part.size(-1)]), r=self.r)
        mask = torch.Tensor(mask)[None, None].to(head_feat.device)
        aux_head_feat_low = torch.fft.ifft2(torch.fft.ifftshift(aux_low_part * mask)).real
        aux_head_feat_hf = aux_head_feat - aux_head_feat_low
        aux_head_feat += aux_head_feat_hf
        # seg = self.decode_head.conv_seg(head_feat)
        aux_seg = self.auxiliary_head.conv_seg(aux_head_feat)
        seg_out = seg + aux_seg
        return x, head_feat, seg_out, None
        
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)
        # # out = self._decode_head_forward_test(x, img_metas)
        # head_feat=self.decode_head._forward_feature(x)
        if self.mode == 'final':
            x, head_feat, seg, edge = self.get_final_feats(img)
        elif self.mode == 'aux':
            x, head_feat, seg, edge = self.get_aux_final_feats(img)
        elif self.mode == 'fused':
            x, head_feat, seg, edge = self.get_fused_final_feats(img)
        else:
            raise NotImplementedError
        # head_feat = self.upsampler(head_feat)
        # initial_seg = self.decode_head.conv_seg(head_feat)
        # out = self.upsampler(head_feat, initial_seg)
        # seg = resize(
        #     input=seg,
        #     scale_factor=4,
        #     mode='nearest')
        # low_part = torch.fft.fftshift(torch.fft.fft2(seg))
        # mask = mask_radial(np.zeros([low_part.size(-2), low_part.size(-1)]), r=self.r)
        # mask = torch.Tensor(mask)[None, None].to(head_feat.device)
        # seg = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
        out = resize(
            input= seg,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out

@SEGMENTORS.register_module()
class EncoderDecoderDGF(EncoderDecoder):
    def __init__(self, r=64, mode='final', *args, **kargs):
        super().__init__(*args, **kargs)
        self.r = r
        self.mode = mode
        self.dgf = nn.Sequential(nn.Conv2d(19, 32, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 19, kernel_size=1, bias=False))

    def get_final_feats(self, img):
        x = self.extract_feat(img)
        head_feat=self.decode_head._forward_feature(x)
        # head_feat=self.auxiliary_head._forward_feature(x)
        seg = self.decode_head.conv_seg(head_feat)
        seg = self.dgf(seg)
        # seg = self.auxiliary_head.conv_seg(head_feat)
        return x, head_feat, seg, None


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x, head_feat, seg, edge = self.get_final_feats(img)

        out = resize(
            input= seg,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out
        
@SEGMENTORS.register_module()
class EncoderDecoderWithContextHead(EncoderDecoder):
    def __init__(self, 
                context_head,
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.context_head = builder.build_head(context_head)
        # del self.context_head.conv_seg
    
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        x = list(x)
        x[-1] = self.context_head._forward_feature(x)
        if self.with_neck:
            x = self.neck(x)
        return x

@SEGMENTORS.register_module()
class EncoderDecoderForVis(EncoderDecoder):
    def __init__(self, 
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.GetSemanticEdge = GetSemanticEdge()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, feats_for_vis = self.extract_feat(img)
        print(img_metas)
        # {'harmful_hfs':harmful_hfs, 'useful_hfs':useful_hfs, 'lower_feats':lower_feats, 'higher_feats':higher_feats, 
                            #  'naive_fused_feats':naive_fused_feats, 'freqfusion_fused_feats':freqfusion_fused_feats}
        harmful_hfs = feats_for_vis['feats_for_vis']
        useful_hfs = feats_for_vis['useful_hfs']
        lower_feats = feats_for_vis['lower_feats']
        higher_feats = feats_for_vis['higher_feats']
        naive_fused_feats = feats_for_vis['naive_fused_feats']
        freqfusion_fused_feats = feats_for_vis['freqfusion_fused_feats']
        
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out

@SEGMENTORS.register_module()
class EncoderDecoderWithDyUP(EncoderDecoder):
    def __init__(self, compress_channel=32, compress_k=1, edge_loss_weight=0, dropout_ratio=0.1, upsampler_mode='imageguide', up2_times=2, 
                use_deformable=False, use_leanable_hpf=False, use_extra_conv3=False,
                use_img=True, use_att=False, 
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.upsampler_mode = upsampler_mode
        # self.GetSemanticEdge = GetSemanticEdge()
        self.edge_loss_weight = edge_loss_weight
        self.use_img = use_img
        self.use_att = use_att
        if use_att:
            self.edge_conv = nn.Conv2d(self.decode_head.channels, 1, kernel_size=3, padding=1)
        if 'carafe' in upsampler_mode:
            self.edge_loss_weight = 0
            self.use_img = False
            self.use_att = False
            print('######## using carafe_seg, set edge_loss_weight to 0')
            print('######## using carafe_seg, set use_img to False')
            print('######## using carafe_seg, set use_att to False')
        if upsampler_mode == 'carafe':
            self.upsampler = CARAFEPack(channels=self.decode_head.channels, scale_factor=2 ** up2_times, up_kernel=3, compressed_channels=compress_channel)
            assert self.edge_loss_weight == 0
            assert self.use_img == False
            assert self.use_att == False
        elif upsampler_mode == 'carafe_seg':
            self.upsampler = CARAFEUpsample(channels=self.decode_head.channels, scale_factor=2 ** up2_times, up_kernel=3, compressed_channels=compress_channel)
            assert self.edge_loss_weight == 0
            assert self.use_img == False
            assert self.use_att == False
        elif upsampler_mode == 'imageguide':
            if compress_channel <= 0:
                self.comp_conv = nn.Identity()
                compress_channel = self.decode_head.channels
            else:
                self.comp_conv = nn.Sequential(
                    nn.Conv2d(self.decode_head.channels, compress_channel, kernel_size=compress_k, padding=compress_k // 2),
                    # nn.SyncBatchNorm(compress_channel),
                    # nn.ReLU(True),
                    # nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
                )
            xavier_init(self.comp_conv, distribution='uniform')
            # channels not applicable
            _channels = (compress_channel + 1) if use_extra_conv3 else (compress_channel + 3)
            self.upsampler = CascadeImageGuideUpsample(n=up2_times, channels=_channels, scale_factor=1, up_kernel=3, compressed_channels=_channels, 
                                use_deformable=use_deformable, use_leanable_hpf=use_leanable_hpf, align_corners=self.align_corners, use_extra_conv3=use_extra_conv3)
        else:
            raise NotImplementedError

        # self.upsampler = DyUpsample(channels=self.decode_head.channels, scale_factor=scale_factor, up_kernel=3, compressed_channels=self.decode_head.channels // 4)
        # self.upsampler = DyUpsample(channels=self.decode_head.channels + 3 * (2 ** up2_times) ** 2, scale_factor=2 ** up2_times, up_kernel=3, compressed_channels=self.decode_head.channels // 4)
        # self.upsampler = DyUpsample(channels=self.decode_head.channels + 3 * (2 ** up2_times) ** 2, scale_factor=2 ** up2_times, up_kernel=3, compressed_channels=compress_channel)
        # self.Temp = nn.parameter.Parameter(torch.Tensor([1.]))
        # self.use_lpr = use_lpr
        # if use_lpr:
            # self.LPR = LocalPixelRelationConv(self.num_classes, kernel_size=3, stride=1, groups=1, dilation=patch_size)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        # self.dct_saver = DCTSaver(save_path='/data3/chenlinwei/code/ResolutionDet/mmseg_exp/utils/Feature_DCT/baseline')
        # self.dct_saver = DCTSaver(save_path='/data3/chenlinwei/code/ResolutionDet/mmseg_exp/utils/Feature_DCT/FreqFusion')

    def get_final_feats(self, img):
        x = self.extract_feat(img)
        # for f in x: self.dct_saver.save(torch_patchwise_dct(f.detach()))
        head_feat=self.decode_head._forward_feature(x)
        if self.upsampler_mode == 'carafe':
            # head_feat = self.upsampler(head_feat)
            head_feat = checkpoint(self.upsampler, head_feat)
            if self.dropout is not None:
                head_feat = self.dropout(head_feat)
            seg = self.decode_head.conv_seg(head_feat)
            edge = self.edge_conv(head_feat)
        elif self.upsampler_mode == 'carafe_seg':
            # compressed_feat = self.comp_conv(head_feat)
            if self.dropout is not None:
                head_feat = self.dropout(head_feat)
            edge = self.edge_conv(head_feat)
            seg = self.decode_head.conv_seg(head_feat)
            # edge_att = resize(input=edge.sigmoid(), size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            seg = self.upsampler(head_feat, seg, img.detach() * 0)
        elif self.upsampler_mode == 'imageguide':
            compressed_feat = self.comp_conv(head_feat)
            if self.dropout is not None:
                head_feat = self.dropout(head_feat)
            seg = self.decode_head.conv_seg(head_feat)
            edge = None
            if self.use_img:
                if self.use_att:
                    edge = self.edge_conv(head_feat)
                    edge_att = resize(input=edge.sigmoid(), size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                    seg = self.upsampler(compressed_feat, seg, img * edge_att.sigmoid())
                else:
                    seg = self.upsampler(compressed_feat, seg, img)
            else:
                seg = self.upsampler(compressed_feat, seg, img.detach() * 0)
        else:
            raise NotImplementedError
        # print(seg.shape)
        return x, head_feat, seg, edge

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)
        # # out = self._decode_head_forward_test(x, img_metas)
        # head_feat=self.decode_head._forward_feature(x)
        x, head_feat, seg, edge = self.get_final_feats(img)
        # head_feat = self.upsampler(head_feat)
        # initial_seg = self.decode_head.conv_seg(head_feat)
        # out = self.upsampler(head_feat, initial_seg)
        out = resize(
            input= seg,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        x, head_feat, seg_logits, edge = self.get_final_feats(img)
        # seg_logits = self.decode_head.conv_seg(head_feat)
        # print(seg_logits.shape)
        # seg_logits = self.upsampler(head_feat, seg_logits)
        # initial_seg = resize(
        #     input=initial_seg,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        losses = dict()
        if self.edge_loss_weight > 0:
            semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
            # b, c, h, w = semantic_edge.shape
            # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
            semantic_edge[semantic_edge > 0] = 1.
            # semantic_edge[semantic_edge <= 0] = 0.0001
            semantic_edge[semantic_edge <= 0] = 0.
            # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            semantic_edge = torch.nn.AdaptiveAvgPool2d(edge.shape[-2:])(semantic_edge)
            b, c, h, w = edge.shape
            # semantic_edge_grid = self.nur.create_grid(semantic_edge, h, w).detach()
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            # losses[f'semantic_edge_loss'] = self.edge_loss_weight * torch.clamp(
            #         F.mse_loss(0.5 * (grid + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
            #     ).mean()
            losses[f'semantic_edge_loss'] = self.edge_loss_weight * F.binary_cross_entropy_with_logits(edge, semantic_edge)
        loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # print(losses)
        # losses = self.AWL(losses)
        # print(self.AWL.params)
        if self.use_awl:losses = self.AWL(losses)
        return losses

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
    
@SEGMENTORS.register_module()
class EncoderDecoderWithDySample(EncoderDecoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.dysampler = DySample(in_channels=self.decode_head.channels, scale=8, style='lp', groups=4, dyscope=True)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        # out = self._decode_head_forward_test(x, img_metas)
        head_feat=self.decode_head._forward_feature(x)
        head_feat = self.dysampler(head_feat)
        out = self.decode_head.cls_seg(head_feat)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        x = self.extract_feat(img)

        losses = dict()

        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        head_feat=self.decode_head._forward_feature(x)
        head_feat = self.dysampler(head_feat)
        seg_logits = self.decode_head.cls_seg(head_feat)
        loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        if self.use_awl:losses = self.AWL(losses)
        return losses
    
@SEGMENTORS.register_module()
class EncoderDecoderWithNUR(EncoderDecoder):
    def __init__(self, 
                patch_size=1, 
                compress_channel=0, 
                dropout_ratio=0.1, 
                use_lpr=False, 
                upsample_mode='pixelshuffle', 
                edge_loss_weight=100, 
                sampler_mode='semantic_edge_avg_psp', 
                padding_mode='replicate', 
                cat_offset=False,
                add_offset=False,
                reverse_mode=None,
                log_dir=None, 
                *args, **kargs):
        super().__init__(*args, **kargs)
        assert patch_size == 1
        self.patch_size = patch_size
        self.upsample_mode = upsample_mode
        # print(sampler_mode)
        self.nur = NonUniformRepresentation(
                    feat_dim=self.decode_head.channels,
                    sampler_mode=sampler_mode, 
                    # padding_mode='reflect', 
                    padding_mode=padding_mode, 
                    stride=1, padding_size=3, fwhm=3, dilation=1, 
                    log_dir=log_dir,
                    use_checkpoint=False, detach=False, align_corners=self.decode_head.align_corners)
        self.GetSemanticEdge = GetSemanticEdge()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        # if compress_channel <= 0:
        #     self.comp_conv = nn.Identity()
        #     self.conv_offset = nn.Conv2d(self.decode_head.channels, self.patch_size ** 2, kernel_size=1)
        # else:
        #     self.comp_conv = nn.Sequential(
        #         nn.Conv2d(self.decode_head.channels, compress_channel, kernel_size=1),
        #         # nn.SyncBatchNorm(compress_channel),
        #         # nn.ReLU(True),
        #         # nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
        #     )
        #     self.conv_offset = nn.Conv2d(compress_channel, self.patch_size ** 2, kernel_size=1)
        self.Temp = nn.parameter.Parameter(torch.Tensor([1.]))
        self.use_lpr = use_lpr
        if use_lpr:
            self.LPR = LocalPixelRelationConv(self.num_classes, kernel_size=3, stride=1, groups=1, dilation=patch_size)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        self.edge_loss_weight = edge_loss_weight
        # del self.decode_head.conv_seg
        self.cat_offset = cat_offset
        self.add_offset = add_offset
        self.reverse_mode = reverse_mode
        if reverse_mode == None: assert self.edge_loss_weight == 0
        
        conv_seg_channel = self.decode_head.channels
        if cat_offset:
            conv_seg_channel += 3
        self.conv_seg = DCNv2(in_channels=conv_seg_channel, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=1)

    def get_offset_grid(self, img, head_feat):
        """Forward function."""
        # if self.dropout is not None:
            # head_feat = self.dropout(head_feat)
        # output = self.conv_offset(self.comp_conv(head_feat))
        # output = nn.PixelShuffle(self.patch_size)(output) * self.Temp
        # output = output.sigmoid()
        # _, _, h, w = output.shape
        # output = output.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w)
        # if self.log_dir is not None:
        #     saliency = output
        #     self.writer.add_images(f"saliency{h}x{w}", ((saliency - saliency.min()) / (saliency.max() - saliency.min())), global_step=None, walltime=None, dataformats='NCHW')
        #     pass
        # output = self.nur.create_grid(output, h, w)
        saliency, grid = self.nur(head_feat)
        return saliency, grid

    def cls_seg(self, feat):
        """Classify each pixel."""
        # if self.dropout is not None:
        #     feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
        
    def get_final_feats(self, img):
        x = self.extract_feat(img)
        # out = self._decode_head_forward_test(x, img_metas)
        head_feat=self.decode_head._forward_feature(x)
        if self.dropout is not None:
            head_feat = self.dropout(head_feat)
        return x, head_feat

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.""" 
        # x = self.extract_feat(img)
        # # out = self._decode_head_forward_test(x, img_metas)
        # head_feat=self.decode_head._forward_feature(x)
        x, head_feat = self.get_final_feats(img)
        # initial_seg = self.decode_head.cls_seg(head_feat)
        saliency, grid = self.get_offset_grid(img, head_feat)
        offset = self.nur.get_offset(grid, False)
        _, _, fh, fw = head_feat.shape
        offset = offset * 0.5
        offset[:, 0] *= fw
        offset[:, 1] *= fh
        # print('saliency offset', offset)
        if self.cat_offset:
            head_feat = torch.cat([head_feat, offset, saliency], dim=1)
        if self.add_offset:
            offset = offset.repeat(1, 3**2, 1, 1)
            initial_seg = self.conv_seg(head_feat, offset)
        else:
            initial_seg = self.conv_seg(head_feat)
        # initial_seg = self.conv_seg(, dim=1))
        _, _, h, w = img.shape
        if self.reverse_mode is None:
            final_seg = resize(
                input=initial_seg,
                # size=img.shape[2:],
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        elif 'tri_interp' in self.reverse_mode:
            final_seg = self.nur.reverse(initial_seg, grid, h, w)
        else:
            raise NotImplementedError
        return final_seg

    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        # x = self.extract_feat(img)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        # seg_logits = self.encode_decode(img, img_metas)
        x, head_feat = self.get_final_feats(img)
        saliency, grid = self.get_offset_grid(img, head_feat)
        offset = self.nur.get_offset(grid, False)
        _, _, fh, fw = head_feat.shape
        offset = offset * 0.5
        offset[:, 0] *= fw
        offset[:, 1] *= fh
        if self.cat_offset:
            head_feat = torch.cat([head_feat, offset, saliency], dim=1)
        if self.add_offset:
            offset = offset.repeat(1, 3**2, 1, 1)
            initial_seg = self.conv_seg(head_feat, offset)
        else:
            initial_seg = self.conv_seg(head_feat)
        # initial_seg = self.conv_seg(, dim=1))
        _, _, h, w = img.shape
        if self.reverse_mode is None:
            seg_logits = resize(
                input=initial_seg,
                # size=img.shape[2:],
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        elif self.reverse_mode == 'tri_interp': # reverse seg using triangle interplotation during training and testing
            seg_logits = self.nur.reverse(initial_seg, grid, h, w)
        elif self.reverse_mode == 'tri_interp_eval': # reverse seg using triangle interplotation only during testing
            seg_logits = initial_seg
        else:
            raise NotImplementedError

        losses = dict()
        if self.edge_loss_weight > 0:
            semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
            # b, c, h, w = semantic_edge.shape
            # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
            semantic_edge[semantic_edge > 0] = 1.
            semantic_edge[semantic_edge <= 0] = 0.0001
            # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            semantic_edge = torch.nn.AdaptiveAvgPool2d(grid.shape[-2:])(semantic_edge)
            b, c, h, w = grid.shape
            semantic_edge_grid = self.nur.create_grid(semantic_edge, h, w).detach()
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            losses[f'semantic_edge_loss'] = self.edge_loss_weight * torch.clamp(
                    F.mse_loss(0.5 * (grid + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
                ).mean()
        if self.log_dir is not None:
            # print(gt_semantic_seg.shape, gt_semantic_seg)
            # print(semantic_edge.shape, semantic_edge.mean(dim=1))
            # semantic_edge_vis = F.interpolate(semantic_edge, , mode='bilinear', align_corners=True)
            # semantic_edge_grid_vis = F.interpolate(semantic_edge_grid, (semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16) , mode='bilinear', align_corners=True)
            semantic_edge_grid_vis = semantic_edge_grid
            semantic_edge_grid_vis = self.nur.get_offset(semantic_edge_grid_vis)
            semantic_edge_grid_vis = (semantic_edge_grid_vis - semantic_edge_grid_vis.min()) / (semantic_edge_grid_vis.max() - semantic_edge_grid_vis.min())
            self.writer.add_images(f"sedge_gridx{h}x{w}", semantic_edge_grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"sedge_gridy{h}x{w}", semantic_edge_grid_vis[:, 1:], global_step=None, walltime=None, dataformats='NCHW')
            # semantic_edge_vis = torch.nn.AdaptiveMaxPool2d((semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16))(semantic_edge)
            semantic_edge_vis = semantic_edge
            self.writer.add_images(f"semantic_edge{h}x{w}", (semantic_edge_vis - semantic_edge_vis.min()) / (semantic_edge_vis.max() - semantic_edge_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
        if self.reverse_mode == 'tri_interp_eval':
            upsampled_grid = F.interpolate(grid, gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=self.decode_head.align_corners)
            gt_semantic_seg = F.grid_sample(gt_semantic_seg.float(), upsampled_grid.permute(0, 2, 3, 1).detach(), mode='nearest', padding_mode="border").long()
            # loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg, grid=upsampled_grid)
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        else:
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # print(losses)
        # losses = self.AWL(losses)
        # print(self.AWL.params)
        if self.use_awl:losses = self.AWL(losses)
        return losses

@SEGMENTORS.register_module()
class EncoderDecoderWithNUR2(EncoderDecoder):
    def __init__(self, 
                patch_size=1, 
                scale_factor=4,
                compress_channel=0, 
                dropout_ratio=-1, 
                use_lpr=False, 
                upsample_mode='shuffle', 
                edge_loss_weight=100, 
                sampler_mode='none',
                padding_mode='replicate', 
                log_dir=None, 
                *args, **kargs):
        super().__init__(*args, **kargs)
        assert patch_size == 1
        assert edge_loss_weight == 0
        self.patch_size = patch_size
        self.upsample_mode = upsample_mode
        self.upsampler = DyUpsample(channels=self.decode_head.channels, scale_factor=scale_factor, up_kernel=5, compressed_channels=self.decode_head.channels // 16)
        self.nur = NonUniformRepresentation(
                    feat_dim=self.decode_head.channels,
                    sampler_mode=sampler_mode, 
                    # padding_mode='reflect', 
                    padding_mode=padding_mode, 
                    stride=1, padding_size=3, fwhm=3, dilation=1, 
                    log_dir=log_dir,
                    use_checkpoint=False, detach=False, align_corners=self.decode_head.align_corners)
        self.edge_loss_weight = edge_loss_weight
        self.GetSemanticEdge = GetSemanticEdge()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        # if compress_channel <= 0:
        #     self.comp_conv = nn.Identity()
        #     self.conv_offset = nn.Conv2d(self.decode_head.channels, self.patch_size ** 2, kernel_size=1)
        # else:
        #     self.comp_conv = nn.Sequential(
        #         nn.Conv2d(self.decode_head.channels, compress_channel, kernel_size=1),
        #         # nn.SyncBatchNorm(compress_channel),
        #         # nn.ReLU(True),
        #         # nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
        #     )
        #     self.conv_offset = nn.Conv2d(compress_channel, self.patch_size ** 2, kernel_size=1)
        self.Temp = nn.parameter.Parameter(torch.Tensor([1.]))
        self.use_lpr = use_lpr
        if use_lpr:
            self.LPR = LocalPixelRelationConv(self.num_classes, kernel_size=3, stride=1, groups=1, dilation=patch_size)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        # self.AWL = AutomaticWeightedLoss(10)
    # def cls_seg(self, feat):
    #     """Classify each pixel."""
    #     if self.dropout is not None:
    #         feat = self.dropout(feat)
    #     output = self.conv_seg(feat)
    #     return output

    def get_offset_grid(self, img, head_feat):
        """Forward function."""
        if self.dropout is not None:
            head_feat = self.dropout(head_feat)
        # output = self.conv_offset(self.comp_conv(head_feat))
        # output = nn.PixelShuffle(self.patch_size)(output) * self.Temp
        # output = output.sigmoid()
        # _, _, h, w = output.shape
        # output = output.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w)
        # if self.log_dir is not None:
        #     saliency = output
        #     self.writer.add_images(f"saliency{h}x{w}", ((saliency - saliency.min()) / (saliency.max() - saliency.min())), global_step=None, walltime=None, dataformats='NCHW')
        #     pass
        # output = self.nur.create_grid(output, h, w)
        # saliency, grid = self.nur(head_feat)
        saliency_logits = self.nur(head_feat, only_logits=True)
        # saliency_logits = self.upsampler(head_feat, saliency_logits)
        saliency_logits = resize(
            input=saliency_logits,
            # size=img.shape[2:],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        _, _, h, w = saliency_logits.shape
        saliency = saliency_logits.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w)
        grid = self.nur.create_grid(saliency, h, w)
        return saliency, grid

    def get_final_feats(self, img):
        x = self.extract_feat(img)
        # out = self._decode_head_forward_test(x, img_metas)
        head_feat=self.decode_head._forward_feature(x)
        return x, head_feat

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)
        # # out = self._decode_head_forward_test(x, img_metas)
        # head_feat=self.decode_head._forward_feature(x)
        x, head_feat = self.get_final_feats(img)
        initial_seg = self.decode_head.cls_seg(head_feat)
        saliency, grid = self.get_offset_grid(img, head_feat)
        _, _, h, w = img.shape
        final_seg = self.nur.sample_with_grid(initial_seg, grid)
        # final_seg = self.nur.reverse(initial_seg, grid, h, w)
        # img_dct = 
        # initial_seg = resize(
        #     input=initial_seg,
        #     # size=img.shape[2:],
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # print(out)
        # out = initial_seg + dct_seg
        return final_seg

    def forward_train(self, img, img_metas, gt_semantic_seg):
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

        # x = self.extract_feat(img)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        # seg_logits = self.encode_decode(img, img_metas)
        x, head_feat = self.get_final_feats(img)
        initial_seg = self.decode_head.cls_seg(head_feat)
        saliency, grid = self.get_offset_grid(img, head_feat)
        _, _, h, w = img.shape
        seg_logits = self.nur.sample_with_grid(initial_seg, grid)
        # seg_logits = self.nur.reverse(initial_seg, grid, h, w)
        # dct_seg = self.dct_seg(img, head_feat)
        # initial_seg = resize(
        #     input=initial_seg,
        #     size=dct_seg.shape[2:],
        #     # size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # seg_logits = initial_seg + dct_seg
        

        losses = dict()
        if self.edge_loss_weight > 0:
            semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
            # b, c, h, w = semantic_edge.shape
            # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
            semantic_edge[semantic_edge > 0] = 1.
            semantic_edge[semantic_edge <= 0] = 0.0001

            # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            semantic_edge = torch.nn.AdaptiveAvgPool2d(grid.shape[-2:])(semantic_edge)
            b, c, h, w = grid.shape
            semantic_edge_grid = self.nur.create_grid(semantic_edge, h, w).detach()
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            losses[f'semantic_edge_loss'] = self.edge_loss_weight * torch.clamp(
                    F.mse_loss(0.5 * (grid + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
                ).mean()
        if self.log_dir is not None:
            # print(gt_semantic_seg.shape, gt_semantic_seg)
            # print(semantic_edge.shape, semantic_edge.mean(dim=1))
            # semantic_edge_vis = F.interpolate(semantic_edge, , mode='bilinear', align_corners=True)
            # semantic_edge_grid_vis = F.interpolate(semantic_edge_grid, (semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16) , mode='bilinear', align_corners=True)
            semantic_edge_grid_vis = semantic_edge_grid
            semantic_edge_grid_vis = self.nur.get_offset(semantic_edge_grid_vis)
            semantic_edge_grid_vis = (semantic_edge_grid_vis - semantic_edge_grid_vis.min()) / (semantic_edge_grid_vis.max() - semantic_edge_grid_vis.min())
            self.writer.add_images(f"sedge_gridx{h}x{w}", semantic_edge_grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"sedge_gridy{h}x{w}", semantic_edge_grid_vis[:, 1:], global_step=None, walltime=None, dataformats='NCHW')
            # semantic_edge_vis = torch.nn.AdaptiveMaxPool2d((semantic_edge.size(-2) // 16, semantic_edge.size(-1) // 16))(semantic_edge)
            semantic_edge_vis = semantic_edge
            self.writer.add_images(f"semantic_edge{h}x{w}", (semantic_edge_vis - semantic_edge_vis.min()) / (semantic_edge_vis.max() - semantic_edge_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
        loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # print(losses)
        # losses = self.AWL(losses)
        # print(self.AWL.params)
        if self.use_awl:losses = self.AWL(losses)
        return losses

@SEGMENTORS.register_module()
class EncoderDecoderWithSaliencySamplerOne(EncoderDecoder):
    def __init__(self, edge_loss_weight=50, log_dir=None, *args, **kargs):
        super().__init__(*args, **kargs)
        # print(args)
        # print(kargs)
        self.GetSemanticEdge = GetSemanticEdge()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.edge_loss_weight = edge_loss_weight


    def get_final_feat(self, img):        
        x, grids, smaps = self.extract_feat(img)
        x = list(x)
        # for j in [3, ]:
        #     # _, _, h, w = x[j-1].shape
        #     _, _, h, w = x[3].shape
        #     # x[3] = self.backbone.SS[j].reverse(x[3], grids[j], h, w)
        #     x[3] = self.backbone.SS[j].reverse(x[3], grids[j], h * 2 - 1, w * 2 -1)
        final_feat = self.decode_head._forward_feature(x)

        """
        # sample label
        grids = [i for i in grids if i != None]
        _, _, h, w = gt_semantic_seg.shape
        for grid in grids:
            # print(grid.shape)
            # b, h, w, 2 -> b, 2, h, w
            grid = F.interpolate(grid.permute(0, 3, 1, 2), (h, w), mode = "bilinear", align_corners=True).permute(0, 2, 3, 1)
            gt_semantic_seg = F.grid_sample(gt_semantic_seg.float(), grid, mode='nearest', align_corners=True).long()
        # """
        
        # """
        # FCN 
        for j in [3, ]:
        # for j in [3, 2, 1]:
            # print(seg_logits.shape)
            # print(grids[j].shape)
            # _, _, h, w = seg_logits.shape
            # _, _, h, w = x[j-1].shape
            _, _, h, w = final_feat.shape
            # grids[j] = F.interpolate(grids[j].permute(0, 3, 1, 2), (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            grid_temp = F.interpolate(grids[j].clone(), (h, w), mode='bilinear', align_corners=True)
            final_feat = self.backbone.SS[j].reverse(final_feat, grid_temp, h * 2 - 1, w * 2 -1)
            # print(seg_logits.shape)
        #"""
        return x, grids, smaps, final_feat

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x, grids, smaps = self.extract_feat(img)
        # # reverse the grid sampling
        # x = list(x)
        # # for j in [3, ]:
        # #     # _, _, h, w = x[j-1].shape
        # #     _, _, h, w = x[3].shape
        # #     # x[3] = self.backbone.SS[j].reverse(x[3], grids[j], h, w)
        # #     x[3] = self.backbone.SS[j].reverse(x[3], grids[j], h * 2 - 1, w * 2 -1)
        # final_feat = self.decode_head._forward_feature(x)

        # # print(out.sigmoid())
        # # out = nn.Softmax2d()(out)
        # # """
        # for j in [3, ]:
        # # for j in [3, 2, 1]:
        #     # _, _, h, w = x[j-1].shape
        #     _, _, h, w = final_feat.shape
        #     # final_feat = self.backbone.SS[j].reverse(final_feat, grids[j], h, w)
        #     final_feat = self.backbone.SS[j].reverse(final_feat, grids[j], h * 2 - 1, w * 2 -1)
        # #"""
        x, grids, smaps, final_feat = self.get_final_feat(img)
        out = self.decode_head.cls_seg(final_feat)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
        
        
    def forward_train(self, img, img_metas, gt_semantic_seg):
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
        x, grids, smaps, final_feat = self.get_final_feat(img)
        seg_logits = self.decode_head.cls_seg(final_feat)
        loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses = dict()
        losses.update(loss_decode)
        # print(losses)
        # saliency_loss = sum([i.std() for i in smaps if i != None])
        # losses['saliency_loss'] = saliency_loss * 1000
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        # print(losses)

        semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
        # b, c, h, w = semantic_edge.shape
        b, c, h, w = grids[3].shape
        # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
        semantic_edge[semantic_edge <= 0] = 0.
        semantic_edge[semantic_edge > 0] = 1.
        semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[3].shape[-2:])(semantic_edge)
        semantic_edge_grid = self.backbone.SS[3].create_grid(semantic_edge, h, w)
        def getMSEloss(pred, target):
            b, c, h, w = target.shape
            if h % 2 == 1:
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=True)
            else:
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
            return F.mse_loss(pred, target)

        losses['semantic_edge_loss_3'] = self.edge_loss_weight * getMSEloss(grids[3], semantic_edge_grid)
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
            self.writer.add_images("semantic_edge", (semantic_edge_vis - semantic_edge_vis.min()) / (semantic_edge_vis.max() - semantic_edge_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
            pass
        # final_feat = self.backbone.SS[j]
        return losses
        
    def extract_feat(self, img):
        """Extract features from images."""
        x, grids, smaps = self.backbone(img)

        if self.log_dir is not None:
            img_vis = F.interpolate(img, (img.size(-2) // 8, img.size(-1) // 8), mode='bilinear', align_corners=True)
            self.writer.add_image("img", (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
            pass

        if self.with_neck:
            raise NotImplementedError
            # x = self.neck(x)
        return x, grids, smaps


@SEGMENTORS.register_module()
class EncoderDecoderWithSaliencySampler(EncoderDecoderWithSaliencySamplerOne):
    def __init__(self, sampler_num=3, 
                 reverse_feat_train=True, 
                 *args, **kargs):
        super().__init__(*args, **kargs)
        self.sampler_num = sampler_num
        self.reverse_feat_train = reverse_feat_train


    def get_final_feat(self, x, grids):
        # CKPT = torch.utils.checkpoint.checkpoint
        # x, grids, smaps = self.extract_feat(img)
        # x, grids, smaps = CKPT(self.extract_feat, img)
        x = list(x)
        final_feat = self.decode_head._forward_feature(x)
        final_feat = self.decode_head.cls_seg(final_feat)

        # for j in [3, 2, 1][:self.sampler_num]:
            # _, _, h, w = final_feat.shape
            # grid_temp = F.interpolate(grids[j].clone(), (h, w), mode='bilinear', align_corners=True)
            # final_feat = self.backbone.SS[j].reverse(final_feat, grid_temp, h * 2 - 1, w * 2 - 1)

        final_grid = grids[-self.sampler_num].clone()
        if self.sampler_num > 1:
            for j in [1, 2, 3][-self.sampler_num + 1:]:
                # grid_temp = F.interpolate(grids[j].detach().clone(), final_grid.shape[-2:], mode='bilinear', align_corners=True)
                # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)
                
        if self.training:
            _, _, h, w = final_feat.shape
            final_grid = F.interpolate(final_grid, (h, w), mode='bilinear', align_corners=self.align_corners)
            if h % 2 == 1:
                final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 2 - 1, w * 2 - 1)
            else:
                final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 2, w * 2)
        else: # eval
            final_feat = F.interpolate(final_feat, final_grid.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            _, _, h, w = final_feat.shape
            # final_grid = F.interpolate(final_grid, (h, w), mode='bilinear', align_corners=True)
            if h % 2 == 1:
                # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 16 - 15, w * 16 - 15)
                final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 8 - 7, w * 8 - 7)
                # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 4 - 3, w * 4 - 3)
                # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 2 - 1, w * 2 - 1)
            else:
                final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 8, w * 8)
            
        return final_feat

    def get_aux_final_seg(self, x, grids, aux_head):
        x = list(x)
        final_feat = aux_head._forward_feature(x)
        final_seg = aux_head.cls_seg(final_feat)
        final_grid = grids[-self.sampler_num].clone()
        if self.sampler_num > 2:
            for j in [1, 2, 3][-self.sampler_num + 1:-1]:
                # grid_temp = F.interpolate(grids[j].detach().clone(), final_grid.shape[-2:], mode='bilinear', align_corners=True)
                # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)
        _, _, h, w = final_seg.shape
        final_grid = F.interpolate(final_grid, (h, w), mode='bilinear', align_corners=self.align_corners)
        if h % 2 == 1:
            final_seg = self.backbone.SS[-2].reverse(final_seg, final_grid, h * 2 - 1, w * 2 - 1)
        else:
            final_seg = self.backbone.SS[-2].reverse(final_seg, final_grid, h * 2, w * 2)
        return final_seg

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, grids, smaps = self.extract_feat(img)
        final_feat = self.get_final_feat(x, grids)
        # out = self.decode_head.cls_seg(final_feat)
        # print(final_feat.shape)
        out = final_feat

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
        
    # def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, grids=None):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     if isinstance(self.auxiliary_head, nn.ModuleList):
    #         for idx, aux_head in enumerate(self.auxiliary_head):
    #             loss_aux = aux_head.forward_train(x, img_metas,
    #                                               gt_semantic_seg,
    #                                               self.train_cfg)
    #             losses.update(add_prefix(loss_aux, f'aux_{idx}'))
    #     else:
    #         loss_aux = self.auxiliary_head.forward_train(
    #             x, img_metas, gt_semantic_seg, self.train_cfg)
    #         losses.update(add_prefix(loss_aux, 'aux'))
    # 
    #     return losses

    def forward_train(self, img, img_metas, gt_semantic_seg):
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
        losses = dict()
        if self.reverse_feat_train:
            x, grids, smaps = self.extract_feat(img)
            final_feat = self.get_final_feat(x, grids)
            seg_logits = final_feat
            # seg_logits = self.decode_head.cls_seg(final_feat)
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
            # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
            # print(loss_decode)
        else:
            raise NotImplementedError
            x, grids, smaps = self.extract_feat(img)
            final_feat = self.get_final_feat(x, grids)
            # x, grids, smaps = CKPT(self.extract_feat, img)
            x = list(x)
            final_feat = self.decode_head._forward_feature(x)
            final_feat = self.decode_head.cls_seg(final_feat)
            seg_logits = F.interpolate(final_feat, gt_semantic_seg.shape[-2:], mode='bicubic', align_corners=True)
            deformed_gt_semantic_seg = gt_semantic_seg.clone()
            for j in [1, 2, 3][-self.sampler_num:]:
                grid_temp = F.interpolate(grids[j].clone(), deformed_gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=True)
                deformed_gt_semantic_seg = F.grid_sample(deformed_gt_semantic_seg.float(), grid_temp.detach().permute(0, 2, 3, 1), mode='nearest', align_corners=True).long()
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)
        
        # auxiliary_head index = 2
        if self.with_auxiliary_head:
            """
            deformed_gt_semantic_seg = gt_semantic_seg.clone()
            for j in [1, 2, 3][-self.sampler_num:-1]:
                grid_temp = F.interpolate(grids[j].clone(), gt_semantic_seg.shape[-2:] , mode='bilinear', align_corners=True)
                deformed_gt_semantic_seg = F.grid_sample(deformed_gt_semantic_seg.float(), grid_temp.detach().permute(0, 2, 3, 1), mode='nearest', align_corners=True).long()
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, deformed_gt_semantic_seg)
            """
            aux_seg = self.get_aux_final_seg(x, grids, self.auxiliary_head)
            loss_aux = self.auxiliary_head.losses(aux_seg, gt_semantic_seg)
            losses.update(add_prefix(loss_aux, f'aux'))
            # losses.update(loss_aux)

        semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
        # b, c, h, w = semantic_edge.shape
        # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
        semantic_edge[semantic_edge > 0] = 1.
        semantic_edge[semantic_edge <= 0] = 0.0001


        for j in [1, 2, 3][-self.sampler_num:]:
            b, c, h, w = grids[j].shape
            if j == [1, 2, 3][-self.sampler_num]:
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
                semantic_edge = torch.nn.AdaptiveAvgPool2d(grids[j].shape[-2:])(semantic_edge)
            else:
                semantic_edge = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(semantic_edge)
                semantic_edge = F.interpolate(semantic_edge, grids[j].shape[-2:], mode='bilinear', align_corners=True)
                # semantic_edge = torch.nn.AdaptiveMaxPool2d(grids[j].shape[-2:])(semantic_edge)
            # semantic_edge_softmax = semantic_edge.clone().view(-1, h * w).softmax(-1).view(b, 1, h, w)
            # print(semantic_edge_softmax.max(), semantic_edge_softmax.min())
            # semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge_softmax, h, w)
            semantic_edge_grid = self.backbone.SS[j].create_grid(semantic_edge, h, w)
            # print(F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none").max(dim=-1))
            losses[f'semantic_edge_loss_{j}'] = self.edge_loss_weight * torch.clamp(
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
                    print(f"Loss {k} is NAN!!!")
                    # losses[k] = losses['semantic_edge_loss_3']
                    raise NotImplementedError
                    # losses[k] = 0.
                    # del losses[k]
        return losses
    

@SEGMENTORS.register_module()
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
        # CKPT = torch.utils.checkpoint.checkpoint
        x, grids, smaps = self.extract_feat(img)
        # x, grids, smaps = CKPT(self.extract_feat, img)
        x = list(x)
        # print(x[-1].shape)
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

    def encode_decode(self, img, img_metas):
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
        
        
    def forward_train(self, img, img_metas, gt_semantic_seg):
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
        # print(self.align_corners)
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
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
        # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        # print(loss_decode)
        losses.update(loss_decode)

        if self.aliasing_loss_weight > 0:
            x_sampleds = smaps['x_sampleds']
            x_sampleds = [i for i in x_sampleds if i != None]
            aliasing_losses = {f'aliasing_loss_{i}':self.AliasingLoss(feat) * self.aliasing_loss_weight for i, feat in enumerate(x_sampleds)} 
            # aliasing_losses = {f'aliasing_loss_{i}':self.AliasingLoss(feat.mean(dim=1, keepdim=True)) for i, feat in enumerate(x_sampleds)}
            # print(aliasing_losses)
            losses.update(aliasing_losses)

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
            loss_aux = self.auxiliary_head.losses(aux_seg, gt_semantic_seg)
            losses.update(add_prefix(loss_aux, f'aux'))

        semantic_edge = self.GetSemanticEdge(gt_semantic_seg)
        # b, c, h, w = semantic_edge.shape
        # semantic_edge_loss_3 = self.GetSemanticEdge.get_loss(smaps[3], semantic_edge)
        semantic_edge[semantic_edge > 0] = 1.
        # semantic_edge[semantic_edge <= 0] = 0.0001 ### ori
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
                # F.mse_loss(0.5 * (grids[j] + 1), 0.5 * (semantic_edge_grid + 1), reduction="none"), min=0, max=0.1
                F.mse_loss(0.5 * (grids[j] + 1)[semantic_edge.repeat(1, 2, 1, 1) > 0.001 + 1e-8], 0.5 * (semantic_edge_grid + 1)[semantic_edge.repeat(1, 2, 1, 1) > 0.001 + 1e-8], reduction="none"), min=0, max=0.1 ###20240318
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
        if self.use_awl:losses = self.AWL(losses)
        return losses

    def extract_feat(self, img):
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

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

class LearnableTriInterp(nn.Module):
    def __init__(self, kernel_size=11, sample_points=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sample_points = sample_points
        # self.offset = nn.Conv2d(in_channels=2, out_channels=sample_points*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.offset_predict = nn.parameter.Parameter(torch.zeros(1, 2, sample_points, kernel_size, kernel_size), requires_grad=True)
        self.kernel_coord_embed = nn.parameter.Parameter(torch.zeros(1, 2, 1, kernel_size, kernel_size), requires_grad=True)
        self.weight_conv = nn.Conv2d(in_channels=sample_points*2, out_channels=sample_points, kernel_size=1, padding=0)

    def forward(self, feature, grid_offset):
        # 计算偏移
        # offset = self.offset(grid_offset)
        b, _, h, w = grid_offset.size()
        _, c, _, _ = feature.size()
        grid_offset = F.unfold(F.pad(grid_offset, pad=[self.kernel_size//2] * 4, mode='reflect'),
                    kernel_size=self.kernel_size, dilation=1, padding=0, stride=1)
        grid_offset = grid_offset.reshape(b, -1, h, w)
        grid_offset = grid_offset.permute(0, 2, 3, 1) # b, h, w, cxkxk
        # print(grid_offset.shape)
        grid_offset = grid_offset.reshape(b, h * w * 2, 1, self.kernel_size, self.kernel_size) # b, h * w * 2, 1, k, k
        offset = (grid_offset.repeat(1, 1, self.sample_points, 1, 1) + self.kernel_coord_embed.repeat(1, h*w, 1, 1, 1)) * self.offset_predict.repeat(1, h*w, 1, 1, 1)
        offset = torch.sum(offset, dim=(-1, -2), keepdim=False)
        offset = offset.reshape(b, h, w, 2, self.sample_points)
        offset = offset.permute(0, 4, 3, 1, 2)
        offset = offset.reshape(b, self.sample_points * 2, h, w)
        # 根据采样坐标计算权重
        weights = self.weight_conv(offset).softmax(dim=1) #b, sample_points, h, w
        
        # 生成采样坐标
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(b, self.sample_points, 1, 1).to(feature.device)  # [b, 2 * sample_points, h, w]
        sample_offset = grid + offset
        sample_offset = sample_offset.reshape(b * self.sample_points, 2, h, w)
        feature = feature.reshape(b, 1, -1, feature.size(-2), feature.size(-1)).repeat(1, self.sample_points, 1, 1, 1)
        feature = feature.reshape(b * self.sample_points, feature.size(-3), feature.size(-2), feature.size(-1))
        sampled_feature = F.grid_sample(feature, sample_offset.permute(0, 2, 3, 1), align_corners=(h % 2 == 1), padding_mode='border').reshape(b, self.sample_points, c, h, w)
        # 加权平均
        weighted_feature = sampled_feature * weights.unsqueeze(2)
        output = torch.sum(weighted_feature, dim=1)
        return output
    
class FeatLearnableTriInterp(nn.Module):
    def __init__(self, kernel_size=11, sample_points=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sample_points = sample_points
        self.offset = nn.Conv2d(in_channels=2, out_channels=sample_points*2, kernel_size=kernel_size, padding=kernel_size//2)
        # self.offset_predict = nn.parameter.Parameter(torch.zeros(1, 2, sample_points, kernel_size, kernel_size), requires_grad=True)
        # self.kernel_coord_embed = nn.parameter.Parameter(torch.zeros(1, 2, 1, kernel_size, kernel_size), requires_grad=True)
        self.weight_conv = nn.Conv2d(in_channels=sample_points*2, out_channels=sample_points, kernel_size=1, padding=0)

    def forward(self, feature, grid_offset):
        # 计算偏移
        # offset = self.offset(grid_offset)
        b, _, h, w = grid_offset.size()
        _, c, _, _ = feature.size()
        grid_offset = F.unfold(F.pad(grid_offset, pad=[self.kernel_size//2] * 4, mode='reflect'),
                    kernel_size=self.kernel_size, dilation=1, padding=0, stride=1)
        grid_offset = grid_offset.reshape(b, -1, h, w)
        grid_offset = grid_offset.permute(0, 2, 3, 1) # b, h, w, cxkxk
        # print(grid_offset.shape)
        grid_offset = grid_offset.reshape(b, h * w * 2, 1, self.kernel_size, self.kernel_size) # b, h * w * 2, 1, k, k
        offset = (grid_offset.repeat(1, 1, self.sample_points, 1, 1) + self.kernel_coord_embed.repeat(1, h*w, 1, 1, 1)) * self.offset_predict.repeat(1, h*w, 1, 1, 1)
        offset = torch.sum(offset, dim=(-1, -2), keepdim=False)
        offset = offset.reshape(b, h, w, 2, self.sample_points)
        offset = offset.permute(0, 4, 3, 1, 2)
        offset = offset.reshape(b, self.sample_points * 2, h, w)
        # 根据采样坐标计算权重
        weights = self.weight_conv(offset).softmax(dim=1) #b, sample_points, h, w
        
        # 生成采样坐标
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(b, self.sample_points, 1, 1).to(feature.device)  # [b, 2 * sample_points, h, w]
        sample_offset = grid + offset
        sample_offset = sample_offset.reshape(b * self.sample_points, 2, h, w)
        feature = feature.reshape(b, 1, -1, feature.size(-2), feature.size(-1)).repeat(1, self.sample_points, 1, 1, 1)
        feature = feature.reshape(b * self.sample_points, feature.size(-3), feature.size(-2), feature.size(-1))
        sampled_feature = F.grid_sample(feature, sample_offset.permute(0, 2, 3, 1), align_corners=(h % 2 == 1), padding_mode='border').reshape(b, self.sample_points, c, h, w)
        # 加权平均
        weighted_feature = sampled_feature * weights.unsqueeze(2)
        output = torch.sum(weighted_feature, dim=1)
        return output


@SEGMENTORS.register_module()
class EncoderDecoderWithSSPRLearnableTriInterp(EncoderDecoderWithSaliencySamplerPixelRelation):
    def __init__(self, 
                offset_kernel_size=11,
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.learnable_tri_interp = nn.ModuleList([
            nn.Identity(),
            LearnableTriInterp(kernel_size=offset_kernel_size, sample_points=3) if self.sampler_num > 2 else nn.Identity(),
            LearnableTriInterp(kernel_size=offset_kernel_size, sample_points=3) if self.sampler_num > 1 else nn.Identity(),
            LearnableTriInterp(kernel_size=offset_kernel_size, sample_points=3) if self.sampler_num > 0 else nn.Identity(),
        ])
        

    def get_aux_final_seg(self, x, grids, aux_head):
        x = list(x)
        final_feat = aux_head._forward_feature(x)
        final_seg = aux_head.cls_seg(final_feat)
        # final_grid = grids[-self.sampler_num].clone()
        # final_seg = self.learnable_tri_interp[-self.sampler_num](final_seg, self.backbone.SS[-self.sampler_num].get_offset(final_grid, cell_encode=False))
        if self.sampler_num > 1:
            for j in [1, 2, 3][-self.sampler_num:-1][::-1]:
                # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners)
                # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)
                final_seg = self.learnable_tri_interp[j](final_seg, self.backbone.SS[j].get_offset(grids[j], cell_encode=False))
                # print(final_seg.shape)
            index = -1
        else:
            index = -1
        # _, _, h, w = final_seg.shape
        # final_grid = F.interpolate(final_grid, (h, w), mode='bilinear', align_corners=self.align_corners)
        # if h % 2 == 1:
        #     final_seg = self.backbone.SS[index].reverse(final_seg, final_grid, 
        #                                                 h * self.aux_up_ratio - self.aux_up_ratio + 1, w * self.aux_up_ratio - self.aux_up_ratio + 1)
        # else:
        #     final_seg = self.backbone.SS[index].reverse(final_seg, final_grid, h * self.aux_up_ratio, w * self.aux_up_ratio)
        return final_seg

    def get_final_feat(self, img):
        # CKPT = torch.utils.checkpoint.checkpoint
        x, grids, smaps = self.extract_feat(img)
        # x, grids, smaps = CKPT(self.extract_feat, img)
        x = list(x)
        # print(x[-1].shape)
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

        final_grid = grids[-1].clone()
        # final_seg = self.learnable_tri_interp[-1](final_seg, self.backbone.SS[-1].get_offset(final_grid, cell_encode=False))
        # print(final_seg.shape)
        if self.reverse_final:
            # print(self.align_corners)
            if self.sampler_num > 0:
                for j in [1, 2, 3][-self.sampler_num:][::-1]:
                    # final_grid = F.grid_sample(final_grid, grid_temp.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True)
                    # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners)
                    # final_grid = F.grid_sample(final_grid, grids[j].permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners)
                    final_seg = self.learnable_tri_interp[j](final_seg, self.backbone.SS[j].get_offset(grids[j], cell_encode=False))
                    # print(final_seg.shape)


            # b, _, h, w =x[0].shape
            # up_scale = (w - 1) // (final_feat.size(-1) - 1)
            # final_feat = F.interpolate(final_feat, final_grid.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # final_feat = F.interpolate(final_feat, x[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
            # _, _, h, w = final_feat.shape
            # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h * 4 - 3 * (h % 2), w * 4 - 3 * (w % 2))
            # print(final_feat.shape, x[0].shape)
            # final_feat = self.backbone.SS[-1].reverse(final_feat, final_grid, h, w)
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

@SEGMENTORS.register_module()
class EncoderDecoderWithSaliencySamplerPixelRelationVis(EncoderDecoderWithSaliencySamplerPixelRelation):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, grids, vis, final_feat, _, _, _ = self.get_final_feat(img) # use only final feat for inference
        # out = self.decode_head.cls_seg(final_feat)
        # print(final_feat.shape)
        return vis
        out = final_feat
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
        
    def simple_test(self, img, img_meta, rescale=True):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        # if rescale:
        #     # support dynamic shape for onnx
        #     if torch.onnx.is_in_onnx_export():
        #         size = img.shape[2:]
        #     else:
        #         size = img_meta[0]['ori_shape'][:2]
        #     seg_logit = resize(
        #         seg_logit,
        #         size=size,
        #         mode='bilinear',
        #         align_corners=self.align_corners,
        #         warning=False)
        return seg_logit

import json
import os

@SEGMENTORS.register_module()
class EncoderDecoderAliasingEval(EncoderDecoder):
    def __init__(self, json_path, max_iter=40000, *args, **kargs):
        super().__init__(*args, **kargs)
        for param in self.parameters():
            # print(param.dim() > 1)
            if param.dim() > 1:
                param.requires_grad = False
        self.data = {
            'acc':[],
            'aux_acc':[],

            'x1':[],
            'x2':[],
            'x3':[],

            'x1_4':[],
            'x2_4':[],
            'x3_4':[],

            'x1_8':[],
            'x2_8':[],
            'x3_8':[],

            'x1_low':[],
            'x2_low':[],
            'x3_low':[],

            'x1_high':[],
            'x2_high':[],
            'x3_high':[],
        }
        self.json_path = json_path
        dir_path = os.path.split(self.json_path)[0]
        self.max_iter = max_iter
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def forward_train(self, *args, **kargs):
        self.eval()
        losses = super().forward_train(*args, **kargs)
        print(losses['decode.acc_seg'])
        self.data['acc'].append(losses['decode.acc_seg'].data.item())
        self.data['aux_acc'].append(losses['aux.acc_seg'].data.item())
        if len(self.data['acc']) % 100 == 0:
            with open(self.json_path, "w") as json_file:
                json.dump(self.data, json_file)
        if len(self.data['acc']) >= self.max_iter:
            exit()
        return losses

    def extract_feat(self, img):
        """Extract features from images."""
        # with torch.no_grad():
        x = self.backbone(img)
        e1_low, e1_hi, e1 = self.calculate_low_frequency_energy_ratio(x[0], index=0+1)
        e2_low, e2_hi, e2 = self.calculate_low_frequency_energy_ratio(x[1], index=1+1)
        e3_low, e3_hi, e3 = self.calculate_low_frequency_energy_ratio(x[2], index=2+1)

        # self.data['x1'].append(e1.data.item())
        # self.data['x2'].append(e2.data.item())
        # self.data['x3'].append(e3.data.item())

        # self.data['x1_low'].append(e1_low.data.item())
        # self.data['x2_low'].append(e2_low.data.item())
        # self.data['x3_low'].append(e3_low.data.item())

        # self.data['x1_high'].append(e1_hi.data.item())
        # self.data['x2_high'].append(e2_hi.data.item())
        # self.data['x3_high'].append(e3_hi.data.item())
        print('energy ratio', e1, e2, e3)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def calculate_low_frequency_energy_ratio(self, tensor, index=None):
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor) ** 2
        # print(freq_domain_energy)
        # 计算低频成分的能量
        b, c, h, w = freq_domain_energy.shape
        low_freq_energy = freq_domain_energy[..., h//2 - h//4:h//2 + h//4, w//2 - w//4:w//2 + w//4].sum((-3, -2, -1)) 
        # print(low_freq_energy)
        # 计算低频成分能量与整个频域能量的比值
        total_energy = freq_domain_energy.sum((-3, -2, -1))  # 计算整个频域的能量
        low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
        self.data[f'x{index}'].append(low_freq_energy_ratio.data.item())
        self.data[f'x{index}_low'].append(low_freq_energy.data.item())
        self.data[f'x{index}_high'].append((total_energy - low_freq_energy).data.item())
        low_freq_energy2 = freq_domain_energy[..., h//2 - h//8:h//2 + h//8, w//2 - w//8:w//2 + w//8].sum((-3, -2, -1)) 
        low_freq_energy3 = freq_domain_energy[..., h//2 - h//16:h//2 + h//16, w//2 - w//16:w//2 + w//16].sum((-3, -2, -1)) 
        low_freq_energy_ratio2 = low_freq_energy2 / (total_energy + 1e-8)
        low_freq_energy_ratio3 = low_freq_energy3 / (total_energy + 1e-8)
        self.data[f'x{index}_4'].append(low_freq_energy_ratio2.data.item())
        self.data[f'x{index}_8'].append(low_freq_energy_ratio3.data.item())

        return low_freq_energy, total_energy - low_freq_energy, low_freq_energy_ratio

@SEGMENTORS.register_module()
class EncoderDecoderAliasingLogging(EncoderDecoder):
    def __init__(self, json_path, *args, **kargs):
        super().__init__(*args, **kargs)
        self.data = {
            'acc':[],
            'aux_acc':[],

            'x1':[],
            'x2':[],
            'x3':[],

            'x1_4':[],
            'x2_4':[],
            'x3_4':[],

            'x1_8':[],
            'x2_8':[],
            'x3_8':[],

            'x1_low':[],
            'x2_low':[],
            'x3_low':[],

            'x1_high':[],
            'x2_high':[],
            'x3_high':[],
        }
        self.json_path = json_path
        dir_path = os.path.split(self.json_path)[0]
        # self.max_iter = max_iter
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def forward_train(self, *args, **kargs):
        # self.eval()
        losses = super().forward_train(*args, **kargs)
        # print(losses['decode.acc_seg'])
        # self.data['acc'].append(losses['decode.acc_seg'].data.item())
        # self.data['aux_acc'].append(losses['aux.acc_seg'].data.item())
        if len(self.data['x3']) % 100 == 0:
            with open(self.json_path, "w") as json_file:
                json.dump(self.data, json_file)
        # if len(self.data['acc']) >= self.max_iter:
            # exit()
        return losses

    def extract_feat(self, img):
        """Extract features from images."""
        # with torch.no_grad():
        x = self.backbone(img)
        e1_low, e1_hi, e1 = self.calculate_low_frequency_energy_ratio(x[0].detach(), index=0+1)
        e2_low, e2_hi, e2 = self.calculate_low_frequency_energy_ratio(x[1].detach(), index=1+1)
        e3_low, e3_hi, e3 = self.calculate_low_frequency_energy_ratio(x[2].detach(), index=2+1)

        # self.data['x1'].append(e1.data.item())
        # self.data['x2'].append(e2.data.item())
        # self.data['x3'].append(e3.data.item())

        # self.data['x1_low'].append(e1_low.data.item())
        # self.data['x2_low'].append(e2_low.data.item())
        # self.data['x3_low'].append(e3_low.data.item())

        # self.data['x1_high'].append(e1_hi.data.item())
        # self.data['x2_high'].append(e2_hi.data.item())
        # self.data['x3_high'].append(e3_hi.data.item())
        print('energy ratio', e1, e2, e3)
        print('energy', e1_hi, e2_hi, e3_hi)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def calculate_low_frequency_energy_ratio(self, tensor, index=None):
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor).mean(dim=0, keepdims=True)
        # print(freq_domain_energy)
        # 计算低频成分的能量
        b, c, h, w = freq_domain_energy.shape
        low_freq_energy = freq_domain_energy[..., h//2 - h//4:h//2 + h//4, w//2 - w//4:w//2 + w//4].sum((-3, -2, -1)) 
        # print(low_freq_energy)
        # 计算低频成分能量与整个频域能量的比值
        total_energy = freq_domain_energy.sum((-3, -2, -1))  # 计算整个频域的能量
        low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
        self.data[f'x{index}'].append(low_freq_energy_ratio.data.item())
        self.data[f'x{index}_low'].append(low_freq_energy.data.item())
        self.data[f'x{index}_high'].append((total_energy - low_freq_energy).data.item())
        low_freq_energy2 = freq_domain_energy[..., h//2 - h//8:h//2 + h//8, w//2 - w//8:w//2 + w//8].sum((-3, -2, -1)) 
        low_freq_energy3 = freq_domain_energy[..., h//2 - h//16:h//2 + h//16, w//2 - w//16:w//2 + w//16].sum((-3, -2, -1)) 
        low_freq_energy_ratio2 = low_freq_energy2 / (total_energy + 1e-8)
        low_freq_energy_ratio3 = low_freq_energy3 / (total_energy + 1e-8)
        self.data[f'x{index}_4'].append(low_freq_energy_ratio2.data.item())
        self.data[f'x{index}_8'].append(low_freq_energy_ratio3.data.item())
        return low_freq_energy, total_energy - low_freq_energy, low_freq_energy_ratio

def GetAliasingMap(tensor, win_size=5, power=2, channel_reduction=4):
    with torch.no_grad():
        b, c, h, w = tensor.shape
        tensor = tensor.reshape(b, c // channel_reduction, channel_reduction, h, w)
        tensor = tensor.sum(-3)
        b, c, h, w = tensor.shape
        tensor = F.unfold(F.pad(tensor, pad=[win_size//2] * 4, mode='reflect'),
                        kernel_size=win_size, dilation=1, padding=0, stride=1)
        tensor = tensor.reshape(b, -1, h, w)
        tensor = tensor.permute(0, 2, 3, 1) # b, h, w, cxkxk
        tensor = tensor.reshape(b, h * w * c, win_size, win_size) # b, h * w * c, k, k
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor) ** power # b, h * w * c, k, k
        freq_domain_energy = freq_domain_energy.reshape(b, h, w, c, win_size, win_size)
        # freq_thres = 1/1.41 * 0.5
        freq_thres = 1/2 * 0.5
        mask = torch.zeros_like(freq_domain_energy, device=freq_domain_energy.device)
        # mask[..., round(win_size/2 - win_size * freq_thres):round(win_size/2 + win_size * freq_thres), round(win_size/2 - win_size * freq_thres):round(win_size/2 + win_size * freq_thres)] = 1.0
        # low_freq_energy = (mask * freq_domain_energy).sum((-3, -2, -1)) 
        low_freq_energy = freq_domain_energy[..., win_size//2 - win_size//4:win_size//2 + win_size//4, win_size//2 - win_size//4:win_size//2 + win_size//4].sum((-3, -2, -1))
        # low_freq_energy = freq_domain_energy[..., round(win_size/2 - win_size * freq_thres):round(win_size/2 + win_size * freq_thres), round(win_size/2 - win_size * freq_thres):round(win_size/2 + win_size * freq_thres)].sum((-3, -2, -1))
        # print(round(win_size/2 - win_size * freq_thres), round(win_size/2 + win_size * freq_thres), round(win_size/2 - win_size * freq_thres), round(win_size/2 + win_size * freq_thres))
        total_energy = freq_domain_energy.sum((-3, -2, -1)) # b, h * w * c
        low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
        hi_freq_energy_ratio = 1.0 - low_freq_energy_ratio
        # print('total_energy', total_energy.shape)
        # print('low_freq_energy_ratio', low_freq_energy_ratio.shape)
        print('hi_freq_energy_ratio', hi_freq_energy_ratio.unique())
    return hi_freq_energy_ratio.squeeze()
    
@SEGMENTORS.register_module()
class EncoderDecoderAliasingHardPixelEval(EncoderDecoder):
    def __init__(self, json_path, max_iter=40000, *args, **kargs):
        super().__init__(*args, **kargs)
        for param in self.parameters():
            # print(param.dim() > 1)
            if param.dim() > 1:
                param.requires_grad = False
        self.json_path = json_path
        dir_path = os.path.split(self.json_path)[0]
        self.dir_path = dir_path
        self.max_iter = max_iter
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path + '/x0/'):
            os.makedirs(dir_path + '/x0/')
        if not os.path.exists(dir_path + '/x1/'):
            os.makedirs(dir_path + '/x1/')
        if not os.path.exists(dir_path + '/x2/'):
            os.makedirs(dir_path + '/x2/')
        if not os.path.exists(dir_path + '/CE/'):
            os.makedirs(dir_path + '/CE/')
        if not os.path.exists(dir_path + '/diff/'):
            os.makedirs(dir_path + '/diff/')
        if not os.path.exists(dir_path + '/pred/'):
            os.makedirs(dir_path + '/pred/')
        if not os.path.exists(dir_path + '/gt/'):
            os.makedirs(dir_path + '/gt/')
        if not os.path.exists(dir_path + '/pred_conf/'):
            os.makedirs(dir_path + '/pred_conf/')

    def forward_train(self, img, img_metas, gt_semantic_seg):
        self.eval()
        # print(img_metas)
        ori_name = img_metas[0]['ori_filename']
        ori_name = os.path.split(ori_name)[1][:-4]
        print(ori_name)
        print(img.shape)

        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        if self.use_awl:losses = self.AWL(losses)
        pred = self.decode_head.forward(x)
        pred = pred.softmax(dim=1)
        pred = torch.max(pred, dim=1)[0]
        # print('pred', pred.shape)
        # print('pred', pred.shape)
        # print('pred', pred)
        seg_res = super().forward_test([img], [img_metas])
        diff = (seg_res[0] == gt_semantic_seg.cpu().numpy().squeeze())
        diff = np.logical_or(diff, gt_semantic_seg.cpu().numpy().squeeze()==255)
        gt_semantic_seg_clone = gt_semantic_seg.detach().clone()
        gt_semantic_seg_clone[gt_semantic_seg==255] = 0
        # print(seg_res[0].shape)
        # print(seg_res[0])
        # print(gt_semantic_seg.cpu().numpy().squeeze().shape)
        # print(gt_semantic_seg.cpu().numpy().squeeze())
        # print(diff)
        # F.cross_entropy()
        # print(losses['decode.acc_seg'])
        # print(results)
        # print(losses['decode.loss_ce'].unique())
        # print(losses['decode.loss_ce'].shape)
        with torch.no_grad():
            np.save(self.dir_path + f'/x0/{ori_name}.npy', GetAliasingMap(x[0]).cpu().numpy())
            np.save(self.dir_path + f'/x1/{ori_name}.npy', GetAliasingMap(x[1]).cpu().numpy())
            np.save(self.dir_path + f'/x2/{ori_name}.npy', GetAliasingMap(x[2]).cpu().numpy())
            # np.save(self.dir_path + f'/x0/{ori_name}.npy', GetAliasingMap(x[0].mean(dim=1, keepdims=True)).cpu().numpy())
            # np.save(self.dir_path + f'/x1/{ori_name}.npy', GetAliasingMap(x[1].mean(dim=1, keepdims=True)).cpu().numpy())
            # np.save(self.dir_path + f'/x2/{ori_name}.npy', GetAliasingMap(x[2].mean(dim=1, keepdims=True)).cpu().numpy())
        np.save(self.dir_path + f'/pred_conf/{ori_name}.npy', pred.squeeze().detach().cpu().numpy())
        np.save(self.dir_path + f'/CE/{ori_name}.npy', losses['decode.loss_ce'].squeeze().detach().cpu().numpy())
        np.save(self.dir_path + f'/diff/{ori_name}.npy', diff)
        np.save(self.dir_path + f'/pred/{ori_name}.npy', seg_res[0].astype(np.int16).squeeze())
        np.save(self.dir_path + f'/gt/{ori_name}.npy', gt_semantic_seg_clone.detach().cpu().numpy().astype(np.int16).squeeze())
        print(self.dir_path)
        # self.data['acc'].append(losses['decode.acc_seg'].data.item())
        # self.data['aux_acc'].append(losses['aux.acc_seg'].data.item())
        # if len(self.data['acc']) % 100 == 0:
        #     with open(self.json_path, "w") as json_file:
        #         json.dump(self.data, json_file)
        # if len(self.data['acc']) >= self.max_iter:
        #     exit()
        return losses
    
@SEGMENTORS.register_module()
class EncoderDecoderWithSaliencySamplerPixelRelationAliasingLogging(EncoderDecoderWithSaliencySamplerPixelRelation):
    def __init__(self, json_path, *args, **kargs):
        super().__init__(*args, **kargs)
        self.data = {
            'acc':[],
            'aux_acc':[],

            'x1_sampled':[],
            'x2_sampled':[],
            'x3_sampled':[],

            'x1_sampled_4':[],
            'x2_sampled_4':[],
            'x3_sampled_4':[],

            'x1_sampled_8':[],
            'x2_sampled_8':[],
            'x3_sampled_8':[],

            'x1_sampled_low':[],
            'x2_sampled_low':[],
            'x3_sampled_low':[],

            'x1_sampled_high':[],
            'x2_sampled_high':[],
            'x3_sampled_high':[],


            'x1':[],
            'x2':[],
            'x3':[],

            'x1_4':[],
            'x2_4':[],
            'x3_4':[],

            'x1_8':[],
            'x2_8':[],
            'x3_8':[],

            'x1_low':[],
            'x2_low':[],
            'x3_low':[],

            'x1_high':[],
            'x2_high':[],
            'x3_high':[],
        }
        self.json_path = json_path
        dir_path = os.path.split(self.json_path)[0]
        # self.max_iter = max_iter
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def forward_train(self, *args, **kargs):
        # self.eval()
        losses = super().forward_train(*args, **kargs)
        # print(losses['decode.acc_seg'])
        # self.data['acc'].append(losses['decode.acc_seg'].data.item())
        # self.data['aux_acc'].append(losses['aux.acc_seg'].data.item())
        if len(self.data['x3']) % 100 == 0:
            with open(self.json_path, "w") as json_file:
                json.dump(self.data, json_file)
        # if len(self.data['acc']) >= self.max_iter:
            # exit()
        return losses

    def extract_feat(self, img):
        """Extract features from images."""
        x, grids, smaps = self.backbone(img)
        e1_low, e1_hi, e1 = self.calculate_low_frequency_energy_ratio(x[0], index=0+1)
        e2_low, e2_hi, e2 = self.calculate_low_frequency_energy_ratio(x[1], index=1+1)
        e3_low, e3_hi, e3 = self.calculate_low_frequency_energy_ratio(x[2], index=2+1)
        print('energy ratio', e1, e2, e3)
        print('energy', e1_hi, e2_hi, e3_hi)

        x_sampleds = smaps['x_sampleds']
        # x_sampleds = [i for i in x_sampleds if i != None]
        e1_low, e1_hi, e1 = self.calculate_low_frequency_energy_ratio(x_sampleds[1], index='1_sampled')
        e2_low, e2_hi, e2 = self.calculate_low_frequency_energy_ratio(x_sampleds[2], index='2_sampled')
        e3_low, e3_hi, e3 = self.calculate_low_frequency_energy_ratio(x_sampleds[3], index='3_sampled')
        # self.data['x1'].append(e1.data.item())
        # self.data['x2'].append(e2.data.item())
        # self.data['x3'].append(e3.data.item())

        # self.data['x1_low'].append(e1_low.data.item())
        # self.data['x2_low'].append(e2_low.data.item())
        # self.data['x3_low'].append(e3_low.data.item())

        # self.data['x1_high'].append(e1_hi.data.item())
        # self.data['x2_high'].append(e2_hi.data.item())
        # self.data['x3_high'].append(e3_hi.data.item())
        print('energy ratio', e1, e2, e3)
        print('energy', e1_hi, e2_hi, e3_hi)

        if self.log_dir is not None:
            img_vis = F.interpolate(img, (img.size(-2) // 8, img.size(-1) // 8), mode='bilinear', align_corners=True)
            self.writer.add_image("img", (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()), global_step=None, walltime=None, dataformats='NCHW')
            pass

        if self.with_neck:
            # x = self.neck(x)
            raise NotImplementedError
        return x, grids, smaps
    
    def calculate_low_frequency_energy_ratio(self, tensor, index=None):
        if tensor == None:
            return 0, 0, 0
        tensor = tensor.detach()
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor).mean(dim=0, keepdims=True)
        # print(freq_domain_energy)
        # 计算低频成分的能量
        b, c, h, w = freq_domain_energy.shape
        low_freq_energy = freq_domain_energy[..., h//2 - h//4:h//2 + h//4, w//2 - w//4:w//2 + w//4].sum((-3, -2, -1)) 
        # print(low_freq_energy)
        # 计算低频成分能量与整个频域能量的比值
        total_energy = freq_domain_energy.sum((-3, -2, -1))  # 计算整个频域的能量
        low_freq_energy_ratio = low_freq_energy / (total_energy + 1e-8)
        self.data[f'x{index}'].append(low_freq_energy_ratio.data.item())
        self.data[f'x{index}_low'].append(low_freq_energy.data.item())
        self.data[f'x{index}_high'].append((total_energy - low_freq_energy).data.item())
        low_freq_energy2 = freq_domain_energy[..., h//2 - h//8:h//2 + h//8, w//2 - w//8:w//2 + w//8].sum((-3, -2, -1)) 
        low_freq_energy3 = freq_domain_energy[..., h//2 - h//16:h//2 + h//16, w//2 - w//16:w//2 + w//16].sum((-3, -2, -1)) 
        low_freq_energy_ratio2 = low_freq_energy2 / (total_energy + 1e-8)
        low_freq_energy_ratio3 = low_freq_energy3 / (total_energy + 1e-8)
        self.data[f'x{index}_4'].append(low_freq_energy_ratio2.data.item())
        self.data[f'x{index}_8'].append(low_freq_energy_ratio3.data.item())
        return low_freq_energy, total_energy - low_freq_energy, low_freq_energy_ratio

@SEGMENTORS.register_module()
class EncoderDecoderWithLabelSR(EncoderDecoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        # print(args)
        # print(kargs)
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
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
        x = self.extract_feat(img)

        losses = dict()
        _, _, h, w = gt_semantic_seg.shape
        # nh, nw = h // 16 + h % 2, w // 16 + w % 2
        nh, nw = (h - 1) * 2 + h % 2, (w - 1) * 2 + w % 2
        gt_semantic_seg = F.interpolate(gt_semantic_seg.float(), (nh, nw), mode='nearest').long()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        # print(losses)
        return losses
        

@SEGMENTORS.register_module()
class EncoderDecoderWithPixelRelation(EncoderDecoder):
    def __init__(self, us_channels=128, relation_kernel=3, *args, **kargs):
        super().__init__(*args, **kargs)
        # print(args)
        # print(kargs)
        # self.al
        # self.temperature = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad=True)
        # self.sigmas = nn.parameter.Parameter(torch.ones(10), requires_grad=True)
        if us_channels == None:
            self.feat_compress = nn.Identity()
            us_channels = kargs['decode_head']['channels']
        else:
            self.feat_compress = nn.Conv2d(kargs['decode_head']['channels'], us_channels, kernel_size=1)

        self.feat_up_convs = nn.ModuleList([
            # nn.Conv2d(in_channels=kargs['channels'], out_channels=us_channels, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=kargs['channels'], out_channels=us_channels, kernel_size=3, padding=1),
            ConvModule(
                # kargs['decode_head']['channels'],
                us_channels,
                us_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                conv_cfg=None,
                norm_cfg=kargs['decode_head']['norm_cfg'],
                act_cfg=dict(type='ReLU', inplace=True)),            
            ConvModule(
                # kargs['decode_head']['channels'],
                us_channels,
                us_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                conv_cfg=None,
                norm_cfg=kargs['decode_head']['norm_cfg'],
                act_cfg=dict(type='ReLU', inplace=True)),
        ])
        # self.relation_conv = nn.ModuleList([
        #     ConvModule(
        #         kargs['decode_head']['channels'],
        #         # us_channels,
        #         us_channels,
        #         kernel_size=3,
        #         padding=1,
        #         dilation=1,
        #         conv_cfg=None,
        #         norm_cfg=kargs['decode_head']['norm_cfg'],
        #         act_cfg=dict(type='ReLU')),            
        # ])

        self.pixelrelation = nn.ModuleList([
            # nn.Sequential(
                # nn.Conv2d(in_channels=kwargs['channels'], out_channels=us_channels, kernel_size=3, padding=1),
                # 
            # ), 
            LocalPixelRelationConv(kargs['decode_head']['channels'], kernel_size=relation_kernel, stride=1, groups=1, dilation=2),
            LocalPixelRelationConv(in_channels=relation_kernel**2, kernel_size=relation_kernel, stride=1, groups=1, dilation=2),
            LocalPixelRelationConv(in_channels=relation_kernel**2, kernel_size=relation_kernel, stride=1, groups=1, dilation=2),
        ])
        # del self.decode_head.cls_conv
        self.decode_head.conv_seg = nn.Conv2d(us_channels, kargs['decode_head']['num_classes'], kernel_size=1)
        # print(self.decode_head.cls_conv)

    def get_upsampled_feats(self, backbone_x, use_checkpoint=False):
        # if use_checkpoint:
        #     CKPT = torch.utils.checkpoint.checkpoint
        #     decoder_x = CKPT(self.decode_head._forward_feature, backbone_x)
        #     decoder_x.requires_grad = True
        #     up_feat = CKPT(self.feat_compress, decoder_x)
        #     # up_feat.requires_grad = True
        # else:
        decoder_x = self.decode_head._forward_feature(backbone_x)
        up_feat = self.feat_compress(decoder_x)

        def _inner_get_upsampled_feats(up_feat):
            up_feats = [up_feat]
            for i in range(len(self.feat_up_convs)):
                up_feat = self.feat_up_convs[i](up_feat)
                _, _, h, w = up_feat.shape
                if h % 2 == 1:
                    up_feat = F.interpolate(up_feat, ((h - 1) * 2 + 1, (w - 1) * 2 + 1), mode='bilinear', align_corners=True)
                else:
                    up_feat = F.interpolate(up_feat, (h * 2, w * 2), mode='bilinear', align_corners=False)
                up_feats.append(up_feat)
            
            # for i in up_feats: print("up_feats", i.requires_grad)
            return tuple(up_feats)

        if use_checkpoint:
            CKPT = torch.utils.checkpoint.checkpoint
            up_feats = CKPT(_inner_get_upsampled_feats, up_feat)
        else:
            up_feats = _inner_get_upsampled_feats(up_feat)
        return decoder_x, tuple(up_feats)

   

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        backbone_x = self.extract_feat(img)
        decoder_x, up_feats = self.get_upsampled_feats(backbone_x)
        up_relation = decoder_x
        # up_relation = self.relation_conv(x)
        up_relations = []
        for i in range(len(self.pixelrelation)):
            if i != (len(self.pixelrelation) - 1):
                up_relation = self.pixelrelation[i](up_relation)
                up_relations.append(up_relation)
                _, _, h, w = up_relation.shape
                if h % 2 == 1:
                    up_relation = F.interpolate(up_relation, ((h - 1) * 2 + 1, (w - 1) * 2 + 1), mode='bilinear', align_corners=True)
                else:
                    up_relation = F.interpolate(up_relation, (h * 2, w * 2), mode='bilinear', align_corners=False)
            else:
                up_relation, final_feat = self.pixelrelation[i](up_relation, up_feats[i])
        # final_feat = 
        # for j in [3, 2]:
        # # for j in [3, 2, 1]:
        #     _, _, h, w = x[j-1].shape
        #     final_feat = self.backbone.SS[j].reverse(final_feat, grids[j], h, w)
        out = self.decode_head.cls_seg(final_feat)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
        
    def forward_train(self, img, img_metas, gt_semantic_seg, use_checkpoint=True):
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
        if use_checkpoint:
            CKPT = torch.utils.checkpoint.checkpoint    
            img.requires_grad = True
            backbone_x = CKPT(self.extract_feat, img)
            decoder_x, up_feats = self.get_upsampled_feats(backbone_x, use_checkpoint=True)
            # decoder_x, up_feats = CKPT(self.get_upsampled_feats, backbone_x)                
        else:
            backbone_x = self.extract_feat(img)
            # decoder_x, up_feats = self.get_upsampled_feats(backbone_x, use_checkpoint=True)
            decoder_x, up_feats = self.get_upsampled_feats(backbone_x, use_checkpoint=False)

        def _inter_get_relations(up_relation, use_checkpoint=True):
            up_relations = []
            refined_feats = []
            # for i in up_feats:
                # print(i.shape)
            for i in range(len(self.pixelrelation)):
                up_relation, refined_feat = self.pixelrelation[i](up_relation, up_feats[i], use_checkpoint=use_checkpoint)
                up_relations.append(up_relation)
                refined_feats.append(refined_feat)
                if i != (len(self.pixelrelation) - 1):
                    _, _, h, w = up_relation.shape
                    if h % 2 == 1:
                        up_relation = F.interpolate(up_relation, ((h - 1) * 2 + 1, (w - 1) * 2 + 1), mode='bilinear', align_corners=True)
                    else:
                        up_relation = F.interpolate(up_relation, (h * 2, w * 2), mode='bilinear', align_corners=False)
            # for i in refined_feats:
            # print(i.shape)
            return up_relations, refined_feats

        up_relation = decoder_x
        up_relations, refined_feats = _inter_get_relations(up_relation, use_checkpoint=use_checkpoint)
        # print(self.decode_head.conv_seg)
        """
        for j in [3, 2]:
        # for j in [3, 2, 1]:
            # print(seg_logits.shape)
            # print(grids[j].shape)
            # _, _, h, w = seg_logits.shape
            _, _, h, w = x[j-1].shape
            # grids[j] = F.interpolate(grids[j].permute(0, 3, 1, 2), (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            # grids[j] = F.interpolate(grids[j], (h, w), mode='bilinear', align_corners=True)
            final_feat = self.backbone.SS[j].reverse(final_feat, grids[j], h, w)
            # print(seg_logits.shape)
        """
        # cal the loss
        losses = dict()
        sigma_counter = 0
        for i in range(len(self.pixelrelation)):
            _, _, h, w = refined_feats[i].shape
            seg_logits = self.decode_head.cls_seg(up_feats[i])
            loss_decode = self.decode_head.losses(seg_logits, gt_semantic_seg)
            refined_seg_logits = self.decode_head.cls_seg(refined_feats[i])
            refined_loss_decode = self.decode_head.losses(refined_seg_logits, gt_semantic_seg)
            for k in loss_decode: 
                if "loss" in k:
                    losses[f'{h}_{k}'] = loss_decode[k] * 0.2
                    # losses[f'{h}_{k}'] = 0.9999 * ( (1 / (self.sigmas[sigma_counter] ** 2 + 1e-8)) * loss_decode[k] * 0.2 + self.sigmas[sigma_counter].log() )
                    sigma_counter += 1
                    # https://blog.csdn.net/qq_41214679/article/details/110750812
                else:
                    losses[f'{h}_{k}'] = loss_decode[k]
                    
            for k in refined_loss_decode:
                if ("loss" in k):
                    if i != (len(self.pixelrelation) - 1):
                        losses[f'refined{h}_{k}'] = 0.2 * refined_loss_decode[k]
                        # losses[f'refined{h}_{k}'] = ( (1 / (self.sigmas[sigma_counter] ** 2 + 1e-8)) * refined_loss_decode[k] + self.sigmas[sigma_counter].log() )
                    else:
                        losses[f'refined{h}_{k}'] = refined_loss_decode[k]
                        # losses[f'refined{h}_{k}'] = (1 / (self.sigmas[sigma_counter] ** 2 + 1e-8)) * refined_loss_decode[k] + self.sigmas[sigma_counter].log()
                    sigma_counter += 1
                else:
                    losses[f'refined{h}_{k}'] = refined_loss_decode[k]

        # print(losses)
        # saliency_loss = sum([i.std() for i in smaps if i != None])
        # losses['saliency_loss'] = saliency_loss * 1000
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                backbone_x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
            # for k in loss_aux:
                # losses[k] = (1 / (self.sigmas[sigma_counter] ** 2 + 1e-8)) * loss_aux[k] + self.sigmas[sigma_counter].log()
                # sigma_counter += 1
        # print(losses)
        return losses

import math
from mmseg.models.backbones.interp2d import FrequencyNorm
@SEGMENTORS.register_module()
class EncoderDecoderWithHook(EncoderDecoder):
    def __init__(self, 
                dir_path=None,
                FN_loss_weight=0.1,
                name_list_for_visual=[
                    # 'FS.freq_weight_conv_list.',
                    # 'conv_mask',
                    # 'conv_offset',
                    # 'conv2',
                    # 'backbone',
                    # 'OMNI_ATT',
                    # 'freqfusions.0',
                    'freqfusions.0.dysampler',
                    'freqfusions.1.dysampler',
                    'freqfusions.2.dysampler',
                    # 'decode_head.convs.0.activate',
                    # 'decode_head.convs.1.activate',
                    # 'decode_head.convs.2.activate',
                    # 'decode_head.convs.3.activate',
                    # 'FS',
                    # 'decode_head.conv_seg',
                                      ],
                name_list_not_for_visual=[
                    # 'layer1.',
                    # 'layer2.',
                    # 'OMNI_ATT.',
                    # 'conv2.',
                    # 'decode_head.',
                    # 'backbone.',
                    # 'layer3.',
                    # 'layer4.',
                    # 'FS.',
                    'freqfusions.0.dysampler.',
                    'freqfusions.1.dysampler.',
                    'freqfusions.2.dysampler.',
                                      ],
                *args, **kargs):
        super().__init__(*args, **kargs)
        # self.GetSemanticEdge = GetSemanticEdge()
        self.dir_path = dir_path
        self.input = []
        self.output = []
        self.hooks = []
        self.FN_loss_weight = FN_loss_weight

        self.name_list_for_visual = name_list_for_visual
        self.name_list_not_for_visual = name_list_not_for_visual
        # self.FN = FrequencyNorm(pooling_size=33 if self.align_corners else 32)
        self.print_hook = True
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.hooks_name = []
    def register_hook(self):
        def hook_fn_forward(module, input, output):
            module_name = str(module)
            # print(module_name)

            # for conv, activate
            # print(len(input))
            # print(len(output))
            # self.input.append(input[:2])
            # for i in output:
            #     if isinstance(i, tuple):
            #         for j in i:print('output', j.shape)
            #     else:
            #         print('output', i.shape)
            # print(torch.std(output[0]))
            # self.output.append(output[1:])
            self.input.append(input)
            self.output.append(output)

        # for name, module in self.named_children():
        # for name, module in self.backbone.named_modules():
        for name, module in self.named_modules():
            if self.print_hook: print(name)
            if any(i in name for i in self.name_list_for_visual) and not any(i in name for i in self.name_list_not_for_visual):
                if self.print_hook: print('*** Register:', name)
                # print('Dilation:', module.dilation)
                self.hooks.append(module.register_forward_hook(hook_fn_forward))
                self.hooks_name.append(name)
        self.print_hook = False

    def remove_hook(self):
        self.input=[]
        self.output=[]
        self.hooks_name=[]
        for i in self.hooks:
            i.remove()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ori_name = img_metas[0]['ori_filename']
        ori_name = os.path.split(ori_name)[1][:-4]
        print(ori_name)
        self.register_hook()
        x = self.extract_feat(img)
        # print('input', self.input)   # 打印该模块的输入
        # print('output', self.output) # 打印该模块的输出
        # out = torch.zeros(1, 150, img.shape[-2], img.shape[-1]).cuda()
        out = self._decode_head_forward_test(x, img_metas)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # print(self.input, self.output)
        DIM=0
        # for i, _in in enumerate(self.input):
        #     if not os.path.exists(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/'):
        #         os.makedirs(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/')
        #     np.save(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.mean(dim=DIM).squeeze().detach().cpu().numpy())

        '''
        for i, _out in enumerate(self.output):
            if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/'):
                os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}')
            # if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/'):
            #     os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}')
            # if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/'):
            #     os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}')
            # if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/'):
            #     os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/')
            # np.save(self.dir_path + f'/dilated_in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/dilated_out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.mean(dim=1).squeeze().detach().cpu().numpy())
            # print('_out', _out[0].shape)
            # print('_out', _out[1].shape)
            print(_out.shape)
            np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out.mean(dim=DIM).squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out[1].mean(dim=DIM).squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out[2].mean(dim=DIM).squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out[3].mean(dim=DIM).squeeze().detach().cpu().numpy())
        #'''
        extra_name=''
        ''' for FreqFusion
        for i, _out in enumerate(self.output):
            if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_hr{extra_name}/'):
                os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_hr{extra_name}/')
            if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_lr{extra_name}/'):
                os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_lr{extra_name}')
            if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr{extra_name}/'):
                os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr{extra_name}/')
            if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr{extra_name}/'):
                os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr{extra_name}/')
            # np.save(self.dir_path + f'/dilated_in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/dilated_out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.mean(dim=1).squeeze().detach().cpu().numpy())
            # print('_out', _out[0].shape)
            # print('_out', _out[1].shape)
            np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_hr{extra_name}/{ori_name}.npy', _out[0][0].mean(dim=DIM).squeeze().detach().cpu().numpy())
            np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_mask_lr{extra_name}/{ori_name}.npy', _out[0][1].mean(dim=DIM).squeeze().detach().cpu().numpy())
            np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr{extra_name}/{ori_name}.npy', _out[1].mean(dim=DIM).squeeze().detach().cpu().numpy())
            np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr{extra_name}/{ori_name}.npy', _out[2].mean(dim=DIM).squeeze().detach().cpu().numpy())
        '''
        # # '''
        # for i, _out in enumerate(self.output):
        #     if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_offset_{extra_name}/'):
        #         os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_offset_{extra_name}/')
        #     if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr_sim_{extra_name}/'):
        #         os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr_sim_{extra_name}/')
        #     if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr_sim_{extra_name}/'):
        #         os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr_sim_{extra_name}/')
        #     # np.save(self.dir_path + f'/dilated_in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
        #     # np.save(self.dir_path + f'/dilated_out_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
        #     # np.save(self.dir_path + f'/in_{self.name_list_for_visual[0]}_{i}/{ori_name}.npy', _in.mean(dim=1).squeeze().detach().cpu().numpy())
        #     # print('_out', _out[0].shape)
        #     # print('_out', _out[1].shape)
        #     np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_offset_{extra_name}/{ori_name}.npy', _out[1].mean(dim=DIM).squeeze().detach().cpu().numpy())
        #     np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_hr_sim_{extra_name}/{ori_name}.npy', _out[2].mean(dim=DIM).squeeze().detach().cpu().numpy())
        #     np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_{i}_lr_sim_{extra_name}/{ori_name}.npy', _out[3].mean(dim=DIM).squeeze().detach().cpu().numpy())
        # # '''
        # '''通用
        print(len(self.input))
        print(len(self.output))
        # print(self.input[0][0].shape)
        # print(self.output[0][0].shape)
        print(self.hooks_name)
        def find_M(N):
            N = int(N)
            sqrt_N = math.sqrt(N // 2)
            M = int(sqrt_N)
            return M
            
        for i, (_in, _out, _name) in enumerate(zip(self.input, self.output, self.hooks_name)):
            if not os.path.exists(self.dir_path + f'/in_{_name}/'):
                os.makedirs(self.dir_path + f'/in_{_name}/')
            if not os.path.exists(self.dir_path + f'/out_{_name}/'):
                os.makedirs(self.dir_path + f'/out_{_name}/')
            print(_name)
            if _in[0].squeeze().dim() == 2:
                _shape = _in[0].squeeze().shape
                print(_shape)
                _hw = find_M(_shape[0])
                _in = _in[0].squeeze().permute(1, 0).reshape(_shape[-1], _hw, _hw * 2)
            else:
                _in = _in[0].squeeze()
            if _out[0].squeeze().dim() == 2:
                _shape = _out[0].squeeze().shape
                print(_shape)
                _hw = find_M(_shape[0])
                _out = _out[0].squeeze().permute(1, 0).reshape(_shape[-1], _hw, _hw * 2)
            else:
                _out = _out[0].squeeze()
            # np.save(self.dir_path + f'/in_{_name}/{ori_name}.npy', _in.detach().cpu().numpy())
            np.save(self.dir_path + f'/out_{_name}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/in_{_name}/{ori_name}.npy', _in[0].squeeze().detach().cpu().numpy())
            # np.save(self.dir_path + f'/out_{_name}/{ori_name}.npy', _out[0].squeeze().detach().cpu().numpy())
        # '''
        '''
        if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer1/'):
            os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer1/')
        if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer2/'):
            os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer2/')
        if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer3/'):
            os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer3/')
        if not os.path.exists(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer4/'):
            os.makedirs(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer4/')
        np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer1/{ori_name}.npy', x[0].mean(dim=DIM).squeeze().detach().cpu().numpy())
        np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer2/{ori_name}.npy', x[1].mean(dim=DIM).squeeze().detach().cpu().numpy())
        np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer3/{ori_name}.npy', x[2].mean(dim=DIM).squeeze().detach().cpu().numpy())
        np.save(self.dir_path + f'/out_{self.name_list_for_visual[0]}_layer4/{ori_name}.npy', x[3].mean(dim=DIM).squeeze().detach().cpu().numpy())
        #'''
        # out = torch.zeros(1, 150, 32, 32).cuda()
        # torch.cuda.empty_cache
        self.remove_hook()
        return out
    
    def register_backward_hook(self):
        def hook_fn_backward(module, input, output):
            # print(self.dir_path)
            # np.save(self.dir_path + f'/CE/{ori_name}.npy', losses['decode.loss_ce'].squeeze().detach().cpu().numpy())
            # print(module)  # 打印模块名，用于区分模块
            # print('input', input[0].shape)   # 打印该模块的输入
            # print('input', input)   # 打印该模块的输入
            # print('output', output.shape) # 打印该模块的输出
            self.input.append(input[0])
            self.output.append(output)

        # for name, module in self.named_children():
        for name, module in self.backbone.named_modules():
            if self.print_hook: print(name)
            if any(i in name for i in self.name_list_for_visual) and not any(i in name for i in self.name_list_not_for_visual):
                if self.print_hook: print('Register:', name)
                # print('Dilation:', module.dilation)
                self.hooks.append(module.register_forward_hook(hook_fn_backward))
        self.print_hook = False

    def forward_train(self, img, img_metas, gt_semantic_seg):
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
            print('only for test ERF')
            self.register_backward_hook()
            x = self.extract_feat(img)
            b, c, h, w = x[3].shape
            center_loss = (x[3][:, :, h//2:h//2 + 1, w//2:w//2 + 1] - (x[3][:, :, h//2:h//2 + 1, w//2:w//2 + 1].detach() + 1)).abs()
            # for i, (_in, _out) in enumerate(zip(self.input, self.output)):
            # print(len(self.output))
            # for i, _out in enumerate(self.output):
                # FN_loss += self.FN(_out).mean()
                # print(self.FN(_out).mean())
                # del _out
            # print(FN_loss)
            self.remove_hook()
            #     np.save(self.dir_path + f'/dilated_in_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
            #     np.save(self.dir_path + f'/dilated_out_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
            # self.input = []
            # [del i for i in self.output]
            # self.output = []
            # print(len(self.output))
            losses = dict()
            losses['center_loss'] = center_loss
            # loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
            # losses.update(loss_decode)

            # if self.with_auxiliary_head:
            #     loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            #     losses.update(loss_aux)
            # if self.use_awl:losses = self.AWL(losses)
            return losses
    
@SEGMENTORS.register_module()
class EncoderDecoderFrequencyNorm(EncoderDecoder):
    def __init__(self, 
                dir_path=None,
                FN_loss_weight=0.1,
                name_list_for_visual=[
                    # 'conv_offset',
                    'conv2',
                                      ],
                name_list_not_for_visual=[
                    'layer1',
                    'layer2',
                    'conv2.',
                                      ],
                *args, **kargs):
        super().__init__(*args, **kargs)
        self.GetSemanticEdge = GetSemanticEdge()
        self.dir_path = dir_path
        self.input = []
        self.output = []
        self.hooks = []
        self.FN_loss_weight = FN_loss_weight

        self.name_list_for_visual = name_list_for_visual
        self.name_list_not_for_visual = name_list_not_for_visual
        self.FN = FrequencyNorm(pooling_size=33 if self.align_corners else 32)
        self.print_hook = True
    
    def register_hook(self):
        def hook_fn_forward(module, input, output):
            # print(self.dir_path)
            # np.save(self.dir_path + f'/CE/{ori_name}.npy', losses['decode.loss_ce'].squeeze().detach().cpu().numpy())
            # print(module)  # 打印模块名，用于区分模块
            # print('input', input[0].shape)   # 打印该模块的输入
            # print('input', input)   # 打印该模块的输入
            # print('output', output.shape) # 打印该模块的输出
            if self.training:
                # self.input.append(input[0])
                self.output.append(output)

        # for name, module in self.named_children():
        for name, module in self.backbone.named_modules():
            if self.print_hook: print(name)
            if any(i in name for i in self.name_list_for_visual) and not any(i in name for i in self.name_list_not_for_visual):
                if self.print_hook: print('Register:', name)
                # print('Dilation:', module.dilation)
                self.hooks.append(module.register_forward_hook(hook_fn_forward))
        self.print_hook = False

    def remove_hook(self):
        self.output=[]
        for i in self.hooks:
            i.remove()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ori_name = img_metas[0]['ori_filename']
        ori_name = os.path.split(ori_name)[1][:-4]
        self.register_hook()
        x = self.extract_feat(img)
        # print('input', self.input)   # 打印该模块的输入
        # print('output', self.output) # 打印该模块的输出
        for i, (_in, _out) in enumerate(zip(self.input, self.output)):
            np.save(self.dir_path + f'/dilated_in_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
            np.save(self.dir_path + f'/dilated_out_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())

        self.input = []
        self.output = []
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # print(out)
        self.remove_hook()
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
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
            # print(img_metas)
            self.register_hook()
            x = self.extract_feat(img)
            FN_loss = 0
            # for i, (_in, _out) in enumerate(zip(self.input, self.output)):
            # print(len(self.output))
            for i, _out in enumerate(self.output):
                FN_loss += self.FN(_out).mean()
                # print(self.FN(_out).mean())
                # del _out
            print(FN_loss)
            self.remove_hook()
            #     np.save(self.dir_path + f'/dilated_in_{i}/{ori_name}.npy', _in.squeeze().detach().cpu().numpy())
            #     np.save(self.dir_path + f'/dilated_out_{i}/{ori_name}.npy', _out.squeeze().detach().cpu().numpy())
            # self.input = []
            # [del i for i in self.output]
            # self.output = []
            # print(len(self.output))
            losses = dict()
            losses['FN_loss'] = FN_loss * self.FN_loss_weight
            loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
                losses.update(loss_aux)
            if self.use_awl:losses = self.AWL(losses)
            return losses