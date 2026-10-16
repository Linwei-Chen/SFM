###################################
# https://github.com/lsa1997/PCAA
###################################
import torch.nn as nn
from torch.nn import functional as F
import torch
from mmseg.core import add_prefix
# from .backbones import get_backbone
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.ops import resize
from ..losses import accuracy

import torch.nn as nn
import torch
import numpy as np
from mmcv.runner import BaseModule
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule

class OhemCrossEntropy2d(nn.Module):
    
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, factor=8, weight=None):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.weight = weight

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != self.ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)

        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=self.ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        
        return select_loss_matrix.mean()


    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

class DSNLoss(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduction='mean', weight=None, ohem=False, thresh=0.7, min_kept=100000):
        super(DSNLoss, self).__init__()
        self.ignore_index = ignore_index
        if ohem:
            self.seg_criterion = OhemCrossEntropy2d(ignore_index, thresh, min_kept, weight=weight)
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        align_corners = (h % 2 == 1)
        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=align_corners)
            loss1 = self.seg_criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=align_corners)
            loss2 = self.aux_criterion(scale_pred, target)

            total_loss = loss1 + loss2*0.4
            loss_dict = {'seg_loss':loss1, 'aux_loss':loss2, 'total_loss':total_loss}
            return loss_dict
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=align_corners)
            loss = self.seg_criterion(scale_pred, target)
            loss_dict = {'total_loss':loss}
            return loss_dict

class JointClsLoss(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduction='mean', weight=None, bins=[(4, 4)], align_corners=None,
                ohem=False, thresh=0.7, min_kept=100000):
        super(JointClsLoss, self).__init__()
        self.align_corners = align_corners
        self.ignore_index = ignore_index
        if ohem:
            self.seg_criterion = OhemCrossEntropy2d(ignore_index, thresh, min_kept, weight=weight)
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = FocalLoss(ignore_index=ignore_index)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label_onehot, bin_size, th=0.01):
        cls_percentage = F.adaptive_avg_pool2d(label_onehot, bin_size)
        cls_label = torch.where(cls_percentage>0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))
        cls_label[(cls_percentage<th)&(cls_percentage>0)] = self.ignore_index
        return cls_label

    def forward(self, preds, target, target_onehot):
        h, w = target.size(1), target.size(2)
        if self.align_corners is None:
            align_corners = (h % 2 == 1)
        else:
            align_corners = self.align_corners
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=align_corners)
        seg_loss = self.seg_criterion(scale_pred, target)
        if len(preds) == 3:
            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=align_corners)
            dsn_loss = self.dsn_criterion(scale_pred, target)

            cls_list = preds[2]
            cls_loss = 0
            for cls_pred, bin_size in zip(cls_list, self.bins):
                cls_gt = self.get_bin_label(target_onehot, bin_size)
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_gt) / len(self.bins)

            total_loss = seg_loss + 0.4*dsn_loss + self.cls_weight*cls_loss
            loss_dict = {'seg_loss':seg_loss, 'aux_loss':dsn_loss, 'cls_loss':cls_loss, 'total_loss':total_loss}
            return loss_dict      
        else:
            cls_list = preds[1]
            cls_loss = 0
            for cls_pred, bin_size in zip(cls_list, self.bins):
                cls_gt = self.get_bin_label(target_onehot, bin_size)
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_gt) / len(self.bins)

            cls_loss = self.cls_weight*cls_loss
            # total_loss = seg_loss + self.cls_weight*cls_loss
            # loss_dict = {'seg_loss':seg_loss, 'cls_loss':cls_loss, 'total_loss':total_loss}
            loss_dict = {'seg_loss':seg_loss, 'cls_loss':cls_loss}
            return loss_dict

class FocalLoss(nn.Module):
    ''' focal loss '''
    def __init__(self, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.crit = nn.BCELoss(reduction='none')

    def binary_focal_loss(self, input, target, valid_mask):
        input = input[valid_mask]
        target = target[valid_mask]
        pt = torch.where(target == 1, input, 1 - input)
        ce_loss = self.crit(input, target)
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        loss = loss.mean()
        return loss
        
    def	forward(self, input, target):
        valid_mask = (target != self.ignore_index)
        K = target.shape[1]
        total_loss = 0
        for i in range(K):
            total_loss += self.binary_focal_loss(input[:,i], target[:,i], valid_mask[:,i])
        return total_loss / K

def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous() # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B,-1,rH,rW,C) # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out

def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out

class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """
    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)
              
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        cam = self.conv_cam(self.dropout(x)) # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam)) # [B, K, bin_num_h, bin_num_w]

        residual = x # [B, C, H, W]
        cam = patch_split(cam, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K) # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH*rW, C) # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats) # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats) # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_h * bin_num_w, K, C]
        
        query = self.proj_query(x) # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        
        aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value) # [B, bin_num_h * bin_num_w, rH * rW, C]
        
        out = out.view(B, -1, rH, rW, value.shape[-1]) # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size) # [B, C, H, W]

        out = residual + self.conv_out(out)
        return out, cls_score
        
# class CAAHead(nn.Module):
@HEADS.register_module()
class CAAHead(BaseDecodeHead):
    """
    Class Activation Attention Head
    """
    def __init__(self, in_channels, channels, num_classes, bins=[(4, 4)], 
                norm_layer=nn.SyncBatchNorm, 
                dropout=False, **kwargs):
        super(CAAHead, self).__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)
        self.num_stages = len(bins)
        self.num_classes = num_classes
        self.module_list = []
        self.loss = JointClsLoss(bins=bins, align_corners = self.align_corners)
        for i in range(self.num_stages):
            self.module_list.append(CAAM(channels, num_classes, bins[i], norm_layer))
        self.module_list = nn.ModuleList(self.module_list)
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            # norm_layer(channels),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True)
            )
        self.convb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # norm_layer(channels),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True)
            )
        self.convc = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels, kernel_size=3, padding=1, bias=False),
            # norm_layer(channels),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True)
        )
        seg_convs = []
        if dropout:
            seg_convs.append(nn.Dropout2d(0.1))
        seg_convs.append(nn.Conv2d(channels, num_classes, kernel_size=1))
        self.conv_seg = nn.Sequential(*seg_convs)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print(len(inputs))
        x = self._transform_inputs(inputs)
        cls_list = []
        residual = x
        out = self.conva(x)
        for i in range(self.num_stages):
            out, cls_score = self.module_list[i](out)
            cls_list.append(cls_score)
        out = self.convb(out)
        out = self.convc(torch.cat([out, residual], dim=1))
        return out, cls_list
        
    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        # x = inputs[-1]
        # print(len(inputs))
        # for i in inputs:
        #     if isinstance(i, torch.Tensor):
        #         print(i[0].shape)
        #     else:
        #         print(i)
                
        x = self._transform_inputs(inputs)
        cls_list = []
        residual = x
        out = self.conva(x)
        for i in range(self.num_stages):
            out, cls_score = self.module_list[i](out)
            cls_list.append(cls_score)
        out = self.convb(out)
        out = self.convc(torch.cat([out, residual], dim=1))
        out = self.cls_seg(out)
        return out, cls_list

    def mask_to_onehot(self, mask):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector

        """
        _mask = [(mask == i) for i in range(self.num_classes)]
        # print(_mask[0].shape)
        mask_onehot = torch.cat(_mask, dim=1).float()
        # print(mask_onehot.unique())
        return mask_onehot

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, cls_list, seg_label, deformed_seg_label):
        """Compute segmentation loss."""
        # loss = dict()
        deformed_seg_label_onehot = self.mask_to_onehot(deformed_seg_label)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # if seg_label.dim
        seg_label = seg_label.squeeze(1)
        # print(seg_logit.shape)
        # print(seg_label.shape)
        loss = self.loss(preds=[seg_logit, cls_list], target=seg_label.squeeze(1), target_onehot=deformed_seg_label_onehot)
        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, cls_list = self.forward(inputs)
        losses = self.losses(seg_logits, cls_list, gt_semantic_seg, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits, cls_list = self.forward(inputs)
        return seg_logits

class Seg_Model(nn.Module):
    def __init__(self, num_classes, criterion=None, norm_layer=nn.BatchNorm2d, 
            bins=(4), aux_loss=False, dropout=False, **kwargs):
        super(Seg_Model, self).__init__()
        feat_num = 512
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.head = CAAHead(self.backbone.deep_channels, feat_num, num_classes, bins, norm_layer, dropout)
        self.use_dsn = aux_loss
        if self.use_dsn:
            dsn_convs = [
                nn.Conv2d(self.backbone.dsn_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True)
                ]
            if dropout:
                dsn_convs.append(nn.Dropout2d(0.1))
            dsn_convs.append(nn.Conv2d(256, num_classes, kernel_size=1))
            self.dsn = nn.Sequential(*dsn_convs)
        self.criterion = criterion

    def forward(self, x, labels=None, labels_onehot=None):
        base_outs = self.backbone(x)
        if len(base_outs) == 1:
            x_dsn = base_outs[0]
            x = base_outs[0]
        else:
            x_dsn = base_outs[-2]
            x = base_outs[-1]
        x, cls_list = self.head(x)
        if self.criterion is not None and labels is not None:
            if self.use_dsn:
                x_dsn = self.dsn(x_dsn)
                outs = [x, x_dsn, cls_list]
            else:
                outs = [x, cls_list]
            return self.criterion(outs, labels, labels_onehot)
        else:
            outs = [x, cls_list]
            return outs