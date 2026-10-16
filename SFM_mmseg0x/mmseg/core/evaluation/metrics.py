# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from mmseg.core.evaluation.f_boundary import eval_mask_boundary
import cv2

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

# https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py#L12
def mask_to_boundary(mask, 
                    #  dilation_ratio=0.02
                     dilation=15
                     ):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    # img_diag = np.sqrt(h ** 2 + w ** 2)
    # dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def custom_total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False,
                              mode = 'biou'
                              ):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    list_bar = tqdm(zip(results, gt_seg_maps))
    list_bar.set_description('Computing IoU')
    # print('Computing IoU.....')
    total_gt_edge_pixel = 1e-8
    total_pred_edge_pixel = 1e-8
    total_tp_pixel = 0
    total_fp_pixel = 0
    total_fn_pixel = 0
    for result, gt_seg_map in list_bar:
        # print(result.shape, gt_seg_map.shape)
        # print(ignore_index)
        # print(type(gt_seg_map))
        pred_edges_1pix = cv2.Laplacian(result.astype(np.uint8), cv2.CV_64F)
        gt_edges_1pix = cv2.Laplacian(gt_seg_map.astype(np.uint8), cv2.CV_64F)
        pred_edges_1pix[pred_edges_1pix!=0] = 1
        gt_edges_1pix[gt_edges_1pix!=0] = 1
        
        # ignore = np.zeros(gt_seg_map.shape)
        # ignore[gt_seg_map==ignore_index] = 1 
        # ignore_3pix = cv2.dilate(ignore, np.ones((3, 3), np.uint8), iterations=1)
        # pred_edges_1pix[ignore_3pix == 1] = 0
        # gt_edges_1pix[ignore_3pix == 1] = 0

        # 创建一个膨胀核
        kernel = np.ones((15, 15), np.uint8)
        # 对提取的边缘进行膨胀运算
        pred_edges = cv2.dilate(pred_edges_1pix, kernel, iterations=1)
        gt_edges = cv2.dilate(gt_edges_1pix, kernel, iterations=1)

        # ignore = cv2.dilate(ignore, kernel, iterations=1)
        # print(np.unique(ignore))
        # pred_edges[ignore == 1] = 0
        # gt_edges[ignore == 1] = 0
        
        union = np.maximum(pred_edges, gt_edges)
        inter = pred_edges + gt_edges
        inter[inter < 2] = 0
        inter[inter == 2] = 1
        # print(np.unique(pred_edges))
        # print(np.unique(pred_edges == gt_edges))
        # print(np.unique(gt_edges))
        if mode == 'biou':
            gt_seg_map[union == 0] = ignore_index # ignore union of gt&pred x-pix wide boundary
        elif mode == 'displacement':
            # gt_seg_map[gt_edges == 0] = ignore_index
            gt_seg_map[inter == 0] = ignore_index  # ignore intersection of gt&pred x-pix wide boundary
            
            fp = pred_edges_1pix + gt_edges
            # fp = pred_edges + gt_edges
            fp[fp == 2] = 0 # correct pred 1pix edge
            fp = fp + pred_edges_1pix
            # fp = fp + pred_edges
            fp[fp == 1] = 0 
            fp[fp == 2] = 1 

            fn = gt_edges_1pix + pred_edges
            # fn = gt_edges + pred_edges
            fn[fn == 2] = 0
            # fn = fn + gt_edges
            fn = fn + gt_edges_1pix
            # fn = fn + gt_edges
            fn[fn == 1] = 0 
            fn[fn == 2] = 1 
            # print(np.unique(fp), np.unique(fn))

            total_gt_edge_pixel += gt_edges_1pix.sum()
            total_pred_edge_pixel += pred_edges_1pix.sum()
            total_gt_edge_pixel += gt_edges.sum()
            total_pred_edge_pixel += pred_edges.sum()
            total_tp_pixel += inter.sum()
            total_fp_pixel += fp.sum()
            total_fn_pixel += fn.sum()
        elif mode == 'FP':
            gt_seg_map[pred_edges == 0] = ignore_index # ignore union of gt&pred x-pix wide boundary
        elif mode == 'FN':
            gt_seg_map[gt_edges == 0] = ignore_index # ignore union of gt&pred x-pix wide boundary
    
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    fp_rate = total_fp_pixel / total_pred_edge_pixel
    fn_rate = total_fn_pixel / total_gt_edge_pixel
    # print(f'fp_rate: {fp_rate}', f'fn_rate: {fn_rate}')
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, fp_rate, fn_rate

def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    list_bar = tqdm(zip(results, gt_seg_maps))
    list_bar.set_description('Computing IoU')
    for result, gt_seg_map in list_bar:
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label

def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


# def get_edge(seg, pix=3):
#     print(seg.shape)
#     e = np.zeros_like(seg)
#     s = np.zeros_like(seg)
#     se = np.zeros_like(seg)
#     e[:,:-1] = seg[:,1:]
#     s[:-1,:] = seg[1:,:]
#     se[:-1,:-1] = seg[1:,1:]

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
            [0, -1, 0], 
            [-1, 4, -1], 
            [0, -1, 0], 
            # [-1, -1, -1], 
            # [-1, 8, -1], 
            # [-1, -1, -1], 
        ]
        self.edge_weight = torch.Tensor(edge_weight)[None, None]

        self.gaussian_weight.requires_grad = False
        self.edge_weight.requires_grad = False
        self.loss = loss

    def forward(self, seg_label, ignore_index, pix=3):
        assert pix >=3
        seg_label = torch.FloatTensor(seg_label)[None, None, :]
        b, c, h, w = seg_label.shape
        seg_label = seg_label.clone()
        seg_label[seg_label==ignore_index] = 0
        # self.gaussian_weight = self.gaussian_weight.to(seg_label.device)
        # self.edge_weight = self.edge_weight.to(seg_label.device)
        # seg_label = nn.ReplicationPad2d(padding=1)(seg_label.float())
        # seg_label = F.conv2d(seg_label, weight=self.gaussian_weight.repeat(c, 1, 1, 1), padding=0, groups=c)
        seg_label = nn.ReplicationPad2d(padding=1)(seg_label)
        semantic_edge = F.conv2d(seg_label, weight=self.edge_weight.repeat(c, 1, 1, 1), padding=0, groups=c)
        semantic_edge[semantic_edge!=0] = 1
        semantic_edge = F.max_pool2d(semantic_edge, kernel_size=pix, padding=pix // 2, stride=1)
        return semantic_edge[0, 0].cpu().numpy()

def eval_metrics_bak(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    all_res = {}
    results = list(results)
    gt_seg_maps = list(gt_seg_maps)
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label,)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)
    for k in ret_metrics: print(k, np.nanmean(ret_metrics[k]))

    #-------------------------------------
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, fp_rate, fn_rate = custom_total_intersect_and_union(
            [i.copy() for i in results], [i.copy() for i in gt_seg_maps], num_classes, ignore_index, label_map,
            reduce_zero_label,
            mode='biou')  ####
    _ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)        
    for k in _ret_metrics: 
        all_res['boundary_' + k] = np.nanmean(_ret_metrics[k])
        print('boundary_' + k, ':', all_res['boundary_' + k])

    #-------------------------------------
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, fp_rate, fn_rate = custom_total_intersect_and_union(
            [i.copy() for i in results], [i.copy() for i in gt_seg_maps], num_classes, ignore_index, label_map,
            reduce_zero_label,
            mode='FP')  ####
    _ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)        
    for k in _ret_metrics: 
        if 'Acc' == k:
            all_res['FP_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('FP_rate', ':', all_res['FP_rate'])
        elif 'aAcc' == k:
            all_res['FP_a_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('FP_a_rate', ':', all_res['FP_a_rate'])

    #-------------------------------------
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, fp_rate, fn_rate = custom_total_intersect_and_union(
            [i.copy() for i in results], [i.copy() for i in gt_seg_maps], num_classes, ignore_index, label_map,
            reduce_zero_label,
            mode='FN')  ####
    _ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)        
    for k in _ret_metrics: 
        if 'Acc' == k:
            all_res['FN_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('FN_rate', ':', all_res['FN_rate'])
        elif 'aAcc' == k:
            all_res['FN_a_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('FN_a_rate', ':', all_res['FN_a_rate'])

    #-------------------------------------
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, fp_rate, fn_rate = custom_total_intersect_and_union(
            [i.copy() for i in results], [i.copy() for i in gt_seg_maps], num_classes, ignore_index, label_map,
            reduce_zero_label,
            mode='displacement')  ####
    _ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)        
    for k in _ret_metrics: 
        if 'Acc' == k:
            all_res['mdisplacement_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('mdisplacement_rate', ':', all_res['mdisplacement_rate'])
        elif 'aAcc' == k:
            all_res['mdisplacement_a_rate'] = 1.0 - np.nanmean(_ret_metrics[k])
            print('mdisplacement_a_rate', ':', all_res['mdisplacement_a_rate'])
    all_res['a_fp_rate'] = fp_rate
    all_res['a_fn_rate'] = fn_rate

    print(all_res)
    return ret_metrics

def _eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    #---------------------------------------------------------
    # print(ret_metrics)
    # metrics
    if 'mBFscore' in metrics:
        metrics = [m for m in metrics if m in 'mBFscore']
        '''
        get_edge = GetSemanticEdge().cuda()
        gt_seg_maps = list(gt_seg_maps)
        # list_bar = tqdm(range(len(gt_seg_maps)))
        # list_bar.set_description('Process gt')
        # for i in list_bar: 
        #     if isinstance(gt_seg_maps[i], str):
        #         gt_seg_maps[i] = mmcv.imread(gt_seg_maps[i], flag='unchanged', backend='pillow')

        list_bar = tqdm(range(len(results)))
        list_bar.set_description('Computing edge')
        for i in list_bar: 
            # print(type(results[i]))
            # break
            edge = get_edge(results[i], ignore_index, pix=3)
            results[i][edge==0] = 0
            edge = get_edge(gt_seg_maps[i], ignore_index, pix=3)
            # print(gt_seg_maps)
            gt_seg_maps[i][edge==0] = 0
            # results[i][edge==0] = ignore_index
            # print(type(results[i]))
        # results
        # gt_seg_maps=
        # print('ignore_index', ignore_index)
        # print(results[0].shape)
        # print(gt_seg_maps[0].shape)
        total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
        ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)
        # print(extra_ret_metrics)
        '''
        pix_thres = [0.00088, 0.001875, 0.00375, 0.005]
        Fpc = {i:np.zeros((num_classes)) for i in pix_thres}
        Fc = {i:np.zeros((num_classes)) for i in pix_thres}
        gt_seg_maps = list(gt_seg_maps)
        list_bar = tqdm(range(len(gt_seg_maps)))
        # thresh=3
        list_bar.set_description('Computing BFscore')
        for i in list_bar: 
            for thresh in pix_thres:
                _Fpc, _Fc = eval_mask_boundary(results[i][np.newaxis, ], gt_seg_maps[i][np.newaxis, ],num_classes, num_proc=4, bound_th=float(thresh))
                Fc[thresh] += _Fc
                Fpc[thresh] += _Fpc
        # del seg_out, edge_out, vi, data
        # logging.info('Threshold: ' + thresh)
        # logging.info('F_Score: ' + str(np.sum(Fpc/Fc)/args.dataset_cls.num_classes))
        # logging.info('F_Score (Classwise): ' + str(Fpc/Fc))
        ret_metrics={}
        ret_metrics['Thresholds'] = pix_thres
        for thresh in pix_thres:
            ret_metrics['mBFscore-{thresh}'] = np.sum(Fpc[thresh]/Fc[thresh]) / num_classes
            ret_metrics['BFscore-{thresh}'] = np.sum(Fpc[thresh]/Fc[thresh])
        print(ret_metrics)
        return ret_metrics
    else:
        # extra_ret_metrics={}
    #---------------------------------------------------------
        total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results, gt_seg_maps, num_classes, ignore_index, label_map,
                reduce_zero_label)
        ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                            total_area_pred_label,
                                            total_area_label, metrics, nan_to_num,
                                            beta)

        return ret_metrics

def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mBFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore' or metric == 'mBFscore' :
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
