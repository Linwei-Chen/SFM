# Copyright (c) OpenMMLab. All rights reserved.
from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F

class GetSemanticEdge(torch.nn.Module):
    def __init__(self, loss=torch.nn.MSELoss()):
        super().__init__()
        gaussian_weight = [
            [1/8, 1/4, 1/8], 
            [1/4, 1/2, 1/4], 
            [1/8, 1/4, 1/8], 
        ]
        self.gaussian_weight = torch.Tensor(gaussian_weight)[None, None]

        edge_weight = [
            [-1, -1, -1], 
            [-1, 8, -1], 
            [-1, -1, -1], 
        ]
        self.edge_weight = torch.Tensor(edge_weight)[None, None]

        self.gaussian_weight.requires_grad = False
        self.edge_weight.requires_grad = False
        self.loss = loss

    def forward(self, seg_label, ignore=255):
        b, c, h, w = seg_label.shape
        seg_label = seg_label.clone()
        seg_label[seg_label==255] = 0
        self.gaussian_weight = self.gaussian_weight.to(seg_label.device)
        self.edge_weight = self.edge_weight.to(seg_label.device)
        seg_label = nn.ReplicationPad2d(padding=1)(seg_label.float())
        seg_label = F.conv2d(seg_label, weight=self.gaussian_weight.repeat(c, 1, 1, 1), padding=0, groups=c)
        seg_label = nn.ReplicationPad2d(padding=1)(seg_label)
        semantic_edge = F.conv2d(seg_label, weight=self.edge_weight.repeat(c, 1, 1, 1), padding=0, groups=c)
        return semantic_edge
    
    def get_loss(self, saliency, semantic_edge, norm='softmax'):
        # if norm == 'softmax':
        semantic_edge = semantic_edge.clone()
        # b, c, h, w = saliency.shape
        # if h % 2 == 1:
        #     saliency = F.interpolate(saliency, (h, w), mode='bilinear', align_corners=True)
        # else:
        #     saliency = F.interpolate(saliency, (h, w), mode='bilinear', align_corners=False)
        semantic_edge[semantic_edge <= 0] = 0.
        semantic_edge[semantic_edge > 0] = 1.
        
        # semantic_edge = F.max_pool2d(semantic_edge, kernel_size=3, padding=1, stride=1)
        semantic_edge = torch.nn.AdaptiveMaxPool2d(saliency.shape[-2:])(semantic_edge)
        b, c, h, w = semantic_edge.shape
        # semantic_edge_norm = semantic_edge.view(-1, h*w).softmax(-1).view(b, c, h, w)
        semantic_edge = semantic_edge / (h * w)
        # semantic_edge = (semantic_edge - semantic_edge.min()) / (semantic_edge.max() - semantic_edge.min())
        # saliency_norm = saliency
        # saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return self.loss(saliency, semantic_edge)