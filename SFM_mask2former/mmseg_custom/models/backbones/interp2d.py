
# from operator import index
from cgi import test
from turtle import forward
import warnings

import math
import cv2
from torch import nn
from torch.autograd import Function
import torch
from torch.utils.checkpoint import checkpoint
import mmseg_custom.models.backbones.spatial.qhull as qhull
import numpy as np
import torch.nn.functional as F
import time
import random
from mmcv.cnn import ConvModule
from torch.utils.tensorboard import SummaryWriter  

# from mmseg.models.backbones.softpool import softpool_cuda
# from mmseg.models.backbones.softpool.SoftPool import soft_pool2d, SoftPool2d

from torch.nn.modules.utils import _triple, _pair, _single

class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = nn.ReplicationPad2d(padding=self.padding)(x)
        # x = nn.ZeroPad2d(padding=self.padding)(x)
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None, force_inplace=False):
        kernel_size = _pair(kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = _pair(stride)
        _, c, h, w = x.size()
        e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

def test_softpool():
    m = SoftPool2d(kernel_size=3, stride=1, padding=1)
    # x = torch.rand(2, 3, 17, 17).cuda()
    x = torch.ones(2, 3, 5, 5).cuda()
    y = m(x) - x
    print(y)
    pass

class Interp2D(nn.Module):
    '''
    New 2d Interpolation in Pytorch
    Reference to scipy.griddata
    Args:
        h, w:  height,width of input
        points: points to interpote shape: [num, 2]
        values:  values of points shape:[num, valuedim]
    return:
       2D interpolate result, shape: [valuedim, h, w]
    '''
    def __init__(self, h, w, add_corner=False):
        super(Interp2D,self).__init__()
        row_coord = np.arange(h).repeat([w]).reshape([h,w])
        col_coord = np.arange(w).repeat([h]).reshape([w,h]).T
        self.coord = np.stack([row_coord, col_coord])
        # print(self.coord.shape)
        self.coord = self.coord.transpose([1,2,0]).reshape([-1,2])
        self.add_corner = add_corner
        self.w = w
        self.h = h
        # if self.add_corner==False:
        #     raise Exception('Now add_corner must be true')

    def forward(self, points, values):
        '''
        notes for gradients: numpy based qhull operations find traingular
        simplices (tri_map --- corner locations) and weights for interpolation,
        tri_map and weights are not derivable, but it's ok, because qhull
        traingular operation is deterministic and we don't need to learn
        parameters for it.

        While gradients still flow because we never put values to cpu, we only
        use tri_map to sample pixels from values, which always on gpu.
        '''
        if self.add_corner:
            points = torch.cat([points.cpu(), torch.Tensor([[0,0], [0, self.w-1],
                                  [self.h-1,0], [self.h-1, self.w-1]]).long()], dim=0)
            values = torch.cat([values, torch.zeros([4,values.shape[1]]).to(values.device)], dim=0)
        else:
            points = points.cpu()
           # Add 4 zeros corner points
        self.tri = qhull.Delaunay(points)
        vdim = values.shape[-1]
        # print('points_shape: {}'.format(points.shape))
        isimplex, weights = self.tri.find_simplex(self.coord, return_c=True)
        # attempt to correct CUDA error: device-side assert triggered
        # which may caused by Points outside the triangulation get the value -1.
        if np.sum(isimplex==-1)>0:
            print('WARNING: {} Points outside the triangulation get the value -1, multiplied by 0\n'.format(np.sum(isimplex==-1)))
            isimplex[isimplex==-1] *= 0
        #the array `weights` is filled with the corresponding barycentric coordinates.
        weights = torch.from_numpy(weights).float().to(values.device)
        # print('isimplex_shape original: {}'.format(isimplex.shape))
        isimplex = torch.from_numpy(isimplex).to(values.device)
        isimplex = isimplex.long()
        isimplex = isimplex.reshape([-1,1])
        # print('isimplex_shape: {}, weights_shape: {}'.format(isimplex.shape, weights.shape))

        # shape: isimplex: [h*w,1]      c: [h,w,c]

        simplices =torch.from_numpy(self.tri.simplices).long().to(values.device)

        tri_map = torch.gather(simplices, dim=0, index=isimplex.repeat([1,3]))
        # print('tri_map max:{}, min{}\n'.format(tri_map.max(),tri_map.min()))
        # print('tri_map_shape: {}, values_shape: {}'.format(tri_map.shape, values.shape))

        value_corr = [torch.gather(values, dim=0, index=tri_map[:,i].
                                    reshape([-1,1]).repeat([1,vdim])) for i in range(3)]
        value_corr = torch.stack(value_corr)
        # print('value_corr_shape: {}'.format(value_corr.shape))
        # print('value_corr have none?: {}'.format(torch.isnan(value_corr).sum()))
        weights = weights.transpose(1,0).unsqueeze(2).repeat([1,1,vdim])
        # print('weights_shape: {}'.format(weights.shape))
        # print('weights have none?: {}'.format(torch.isnan(weights).sum()))
        # print('weights_dtype: {}, value_corr_dtype: {}'.format(weights.dtype, value_corr.dtype))
        out = torch.mul(value_corr, weights).sum(dim=0)
        # print('out_shape: {}'.format(out.shape))
        return out.reshape([self.h, self.w, vdim]).permute(2,0,1)


def test_1():
    interp2d = Interp2D(10,10)
    points = torch.rand([10,2])*10
    corner = torch.FloatTensor([[0, 0], [0, 1], [1, 1], [1, 0]]) * 9
    points[0, :], points[1, :], points[2, :], points[3, :]  = corner[0], corner[1], corner[2], corner[3]
    print(points.shape)
    values = torch.rand([10,1])
    values[2] = 1
    t0 = time.perf_counter()
    out = interp2d(points.cuda(), values.cuda())
    print(time.perf_counter() - t0)
    print('out shape', out.shape)
    # print('points\n', points)
    # print('values\n', values)
    print(out)

def test_2():
    interp2d = Interp2D(100,100)
    points = torch.rand([100,2]) * 100
    corner = torch.FloatTensor([[0, 0], [0, 1], [1, 1], [1, 0]]) * 99
    points[0, :], points[1, :], points[2, :], points[3, :]  = corner[0], corner[1], corner[2], corner[3]
    print(points.shape)
    values = torch.rand([100, 2048])
    # values[2, :] = 1
    t0 = time.perf_counter()
    # out = interp2d(points.cuda(), values.cuda())
    out = interp2d(points, values)
    print(time.perf_counter() - t0)
    print('out shape', out.shape)
    # print('points\n', points)
    # print('values\n', values)
    print(out)


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm ** 2)


def makeGaussiantemplete(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return -4 * np.log(2) * ((x-x0)**2 + (y-y0)**2)


class LearnableGaussian(nn.Module):
    def __init__(self, padding_size=5, fwhm=3, requires_grad=True):
        super().__init__()
        gaussian_templete = torch.FloatTensor(makeGaussiantemplete(2 * padding_size + 1))[None, None] # -4 * np.log(2) * ((x-x0)**2 + (y-y0)**2)
        self.gaussian_templete = gaussian_templete
        self.sigma = torch.nn.parameter.Parameter(torch.FloatTensor([fwhm]))
        self.gaussian_templete.requires_grad = True
        self.sigma.requires_grad = requires_grad

    # def weight(self):
        # return torch.exp(self.gaussian_templete / self.sigma ** 2)

    def forward(self, x):
        """
        for learnable sigma
        """        
        # print(self.gaussian_templete.shape)
        # print(self.gaussian_templete.requires_grad)
        # print(self.sigma.requires_grad)
        # print(self.gaussian_templete)
        print(x.shape[2:], self.sigma)
        # print(self.gaussian_templete)
        # print(torch.exp(self.gaussian_templete / self.sigma ** 2).requires_grad)
        # print(torch.exp(self.gaussian_templete / self.sigma ** 2).shape)
        # print(torch.exp(self.gaussian_templete / self.sigma ** 2))
        self.gaussian_templete = self.gaussian_templete.to(x.device)
        self.sigma = self.sigma.to(x.device)
        # print(self.gaussian_templete.device, self.sigma.device)
        self.weight = torch.exp(self.gaussian_templete / self.sigma ** 2)
        # print(self.weight)
        return F.conv2d(x, weight=self.weight, stride=1, padding=0)

class LaplacianConv(nn.Module):
    """
    Learnable Low Pass Filter
    """
    def __init__(self, channels=3, stride=2, padding=1, requires_grad=False):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        # kernel = [
        #     [0, 0., 0],
        #     [0, 1., 0],
        #     [0, 0., 0],
        # ]
        kernel = [
            [-1/8., -1/8., -1/8.],
            [-1/8., 1., -1/8.],
            [-1/8., -1/8., -1/8.],
        ]
        # kernel = [
        #     [1/16., 1/8., 1/16.],
        #     [1/8., 1/4., 1/8.],
        #     [1/16., 1/8., 1/16.],
        # ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1).contiguous()
        self.weight = nn.Parameter(data=kernel, requires_grad=requires_grad)
 
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        # print(self.weight)
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels, stride=self.stride)
        return x

class LHPFConv3(nn.Module):
    """
    可学习的高通滤波器 (Learnable High Pass Filter, LHPF)，通过3x3卷积实现。

    该模块使用深度可分离卷积（depthwise convolution），其卷积核是可学习的。
    通过在Softmax函数归一化一个基础卷积核，使其成为一个低通滤波器（平滑/模糊核）。
    最终的高通效果是通过从一个单位矩阵核（identity kernel）中减去这个可学习的低通核来实现的。
    这个过程可以理解为：高通信号 = 原始信号 - 低通信号。
    """
    def __init__(self, channels: int = 3, stride: int = 2, padding: int = 1, residual: bool = False):
        """
        初始化函数。

        参数:
            channels (int): 输入特征图的通道数。
            stride (int): 卷积步长。
            padding (int): 卷积填充。
        """
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding

        # 1. 初始化用于学习低通滤波器的基础卷积核。
        #    这个核的权重将在前向传播中通过Softmax进行归一化。
        #    初始化为1有助于模型从一个平均（低通）行为开始学习。
        lpf_kernel_base = torch.tensor([
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]
        ], dtype=torch.float32)
        
        # 为每个通道重复此基础核，以构建深度可分离卷积的权重。
        # 形状变为: (channels, 1, 3, 3)
        lpf_kernel_base = lpf_kernel_base.unsqueeze(0).unsqueeze(0)
        lpf_kernel_base = lpf_kernel_base.repeat(self.channels, 1, 1, 1)

        # 将其注册为模型的可学习参数。
        self.weight = nn.Parameter(data=lpf_kernel_base, requires_grad=True)

        # 2. 定义用于高通计算的单位矩阵核。
        #    该卷积核在卷积操作中可以返回中心像素的原始值。
        identity_kernel = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ], dtype=torch.float32)
        
        # 将单位矩阵核注册为不可训练的缓冲区（buffer）。
        # 这可以确保它能随模型自动移动到正确的设备（如GPU），
        # 但又不会被优化器视为需要更新的参数。
        if residual: identity_kernel = identity_kernel * 2 
        self.register_buffer('identity_kernel', identity_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        """
        # 问题修复：确保输入是4D张量 (B, C, H, W)。
        # 如果传入的是3D张量 (C, H, W)，则为其增加一个批次维度。
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # --- 步骤 1: 创建可学习的低通滤波器核 ---
        # 重塑 -> 应用Softmax -> 重塑回原状。
        # Softmax确保核的权重总和为1，使其表现得像一个加权平均（模糊）滤波器。
        weight_softmax = self.weight.reshape(self.channels, 1, -1).softmax(dim=-1)
        low_pass_kernel = weight_softmax.reshape(self.channels, 1, 3, 3)

        # --- 步骤 2: 创建高通滤波器核 ---
        # 通过从单位矩阵核中减去低通核来得到高通核。
        # self.identity_kernel 会被自动广播以匹配 low_pass_kernel 的形状。
        high_pass_kernel = self.identity_kernel - low_pass_kernel

        # --- 步骤 3: 应用卷积 ---
        # 使用计算出的高通核进行深度可分离卷积。
        # `groups=self.channels` 确保每个输入通道都与其对应的卷积核进行运算。
        output = F.conv2d(x, high_pass_kernel, 
                          padding=self.padding, groups=self.channels, stride=self.stride)
        
        return output

class LHPFConv(nn.Module):
    """
    Learnable High Pass Filter
    """
    def __init__(self, channels=3, kernel_size=5, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = torch.rand(self.channels, 1, kernel_size, kernel_size)
        # in_ch, out_ch,
        # kernel = kernel.repeat(self.channels, 1, 1, 1).contiguous()
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_in')

        identity_kernel = torch.zeros(self.channels, 1, kernel_size, kernel_size)
        identity_kernel[:, :, kernel_size // 2, kernel_size // 2] = 1.
        self.identity_kernel = nn.Parameter(data=identity_kernel, requires_grad=False)
        self.kernel_size = kernel_size
 
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.identity_kernel - self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, self.kernel_size, self.kernel_size), 
                    padding=self.padding, groups=self.channels, stride=self.stride)
        return x

def test_makeGaussian():
    # kernel = makeGaussian(size = 5, fwhm = 1)
    # print(kernel)
    x = torch.rand(2, 1, 16, 16)
    m = LearnableGaussian()
    print(m(x))
    print(m.weight)


class LocalCosSim(nn.Module):
    def __init__(self, in_channels=64, mid_channels=0, neighbor_size=3, stride=1, dilation=1, eps=1e-8):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0) if mid_channels > 0 else nn.Identity(),
            nn.SyncBatchNorm(mid_channels) if mid_channels > 0 else nn.SyncBatchNorm(in_channels),
        )
        self.cossim = nn.CosineSimilarity(dim=1)
        self.pad = neighbor_size // 2
        self.neighbor_size = neighbor_size
        self.stride = stride
        self.dilation = dilation
        self.eps = eps
        pass

    def forward(self, x):
        feat = self.compress(x)
        b, c, h, w = feat.shape
        featy = F.pad(feat, pad=[self.pad] * 4, mode='reflect')
        featy = F.unfold(featy, kernel_size=(self.neighbor_size, self.neighbor_size), stride=self.stride, dilation=self.dilation)
        featy = featy.reshape(b, c, self.neighbor_size, self.neighbor_size, h, w)
        feat = feat.repeat(1, self.neighbor_size ** 2, 1, 1).reshape(b, c, self.neighbor_size, self.neighbor_size, h, w)
        cosim = self.cossim(feat, featy).reshape(b, self.neighbor_size ** 2, h, w)
        return cosim

def test_LocalCosSim():
    x = torch.rand(1, 128, 4, 4) * 4
    x[:, :, 0, 0] += 10
    m = LocalCosSim(128, 16)
    print(m(x).shape)

class PSP_Module(nn.Module):
    def __init__(self, in_channels, out_channels, bin):
        super(PSP_Module, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(bin)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv = ConvModule(
        #     in_channels,
        #     out_channels,
        #     kernel_size=1,
        #     padding=0,
        #     dilation=1,
        #     conv_cfg=None,
        #     # norm_cfg=dict(type='BN', requires_grad=True),
        #     norm_cfg=dict(type='SyncBN', requires_grad=True),
        #     act_cfg=dict(type='ReLU', inplace=True)
        #     # act_cfg=None
        # )

    def forward(self, x):
        size = x.size()
        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size[2:], mode='bilinear', align_corners=True)
        return x

class PSP(nn.Module):
    def __init__(self, in_channels, out_channels, bins = [1, 2, 3, 7]):
        super(PSP, self).__init__()
        self.psp1 = PSP_Module(in_channels, out_channels, bins[0])
        self.psp2 = PSP_Module(in_channels, out_channels, bins[1])
        self.psp3 = PSP_Module(in_channels, out_channels, bins[2])
        self.psp4 = PSP_Module(in_channels, out_channels, bins[3])

    def forward(self, x):
        x1 = self.psp1(x)
        x2 = self.psp2(x)
        x3 = self.psp3(x)
        x4 = self.psp4(x)
        out = torch.cat([x, x1, x2, x3, x4], dim=1)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

from multiprocessing import Pool
from functools import partial
def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))
    # return list(map(list, zip(*map_results)))
    # print(map_results)
    return map_results

import torch
import torch.nn.functional as F

def optimized_tiled_non_uniform_upsample(sampled_feat, non_uniform_grid, target_h, target_w, k=3, power=2, tile_size=64, search_margin=8):
    """
    使用分块策略优化的非均匀上采样，适用于高分辨率图像。

    Args:
        sampled_feat (torch.Tensor): 非均匀采样的特征图。
                                     形状: [B, C, H_in, W_in]
        non_uniform_grid (torch.Tensor): 用于采样的非均匀网格坐标，值在 [-1, 1] 之间。
                                         形状: [B, H_in, W_in, 2]
        target_h (int): 目标均匀网格的高度。
        target_w (int): 目标均匀网格的宽度。
        k (int): k-NN插值中邻居点的数量。
        power (int): 逆距离加权的幂次。
        tile_size (int): 将目标网格划分的小块的大小。
        search_margin (int): 在非均匀网格上确定搜索区域时向外扩展的像素数，确保能覆盖到所有相关的邻居点。

    Returns:
        torch.Tensor: 在均匀网格上上采样后的特征图。形状: [B, C, target_h, target_w]
    """
    B, C, H_in, W_in = sampled_feat.shape
    device = sampled_feat.device

    # 1. 准备源点数据（非均匀网格），将其展平以便索引
    src_feat_flat = sampled_feat.view(B, C, -1).permute(0, 2, 1)  # 形状: [B, H_in*W_in, C]
    src_grid_flat = non_uniform_grid.view(B, -1, 2)              # 形状: [B, H_in*W_in, 2]

    # 2. 初始化一个全零的目标输出图像
    output_image = torch.zeros(B, C, target_h, target_w, device=device)

    # 3. 对目标网格进行分块遍历
    for i in range(0, target_h, tile_size):
        for j in range(0, target_w, tile_size):
            # 定义当前目标块的边界
            i_end = min(i + tile_size, target_h)
            j_end = min(j + tile_size, target_w)
            tile_h, tile_w = i_end - i, j_end - j

            # 创建当前块的均匀目标网格坐标
            y_coords = torch.linspace(-1 + 2 * i / target_h, -1 + 2 * i_end / target_h, tile_h, device=device)
            x_coords = torch.linspace(-1 + 2 * j / target_w, -1 + 2 * j_end / target_w, tile_w, device=device)
            # grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
            tgt_tile_grid_flat = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2).expand(B, -1, -1) # 形状: [B, tile_h*tile_w, 2]

            # 4. 确定局部搜索窗口
            # 找到大致落在当前目标块区域内的非均匀采样点
            center_x, center_y = j + tile_w / 2, i + tile_h / 2
            # 将中心点坐标从 [0, W] [0, H] 范围转换到 [-1, 1] 范围
            center_x_norm = (center_x / target_w) * 2 - 1
            center_y_norm = (center_y / target_h) * 2 - 1
            
            # 计算搜索半径，并加上安全边距
            radius_x = (tile_w / target_w) + (search_margin / W_in)
            radius_y = (tile_h / target_h) + (search_margin / H_in)

            # 筛选出搜索窗口内的源点
            search_indices = (src_grid_flat[..., 0] > center_x_norm - radius_x) & \
                             (src_grid_flat[..., 0] < center_x_norm + radius_x) & \
                             (src_grid_flat[..., 1] > center_y_norm - radius_y) & \
                             (src_grid_flat[..., 1] < center_y_norm + radius_y)
            
            # 如果某个batch item的搜索窗为空，则跳过
            if not torch.any(search_indices):
                continue
            
            # 5. 在局部窗口内执行k-NN插值
            # 注意：这里我们为每个batch item单独处理，因为搜索结果可能不同
            for b in range(B):
                b_indices = search_indices[b]
                if not torch.any(b_indices):
                    continue

                local_src_grid = src_grid_flat[b, b_indices]
                
                # 局部距离计算
                dists = torch.cdist(tgt_tile_grid_flat[b], local_src_grid)
                
                # 确保k不大于窗口内的点数
                num_local_points = local_src_grid.shape[0]
                current_k = min(k, num_local_points)

                # 局部k-NN
                topk_dists, topk_local_indices = torch.topk(dists, k=current_k, dim=-1, largest=False)

                # 获取对应的特征
                local_src_feat = src_feat_flat[b, b_indices]
                neighbor_feats = local_src_feat[topk_local_indices] # 形状: [tile_h*tile_w, k, C]
                
                # 计算逆距离权重
                weights = 1.0 / (topk_dists.pow(power) + 1e-8)
                weights = weights / torch.sum(weights, dim=-1, keepdim=True)
                
                # 加权插值
                interpolated_feats = torch.sum(neighbor_feats * weights.unsqueeze(-1), dim=1)
                
                # 将结果填充到输出图像的对应位置
                output_image[b, :, i:i_end, j:j_end] = interpolated_feats.permute(1, 0).view(C, tile_h, tile_w)

    return output_image

class SaliencySampler(nn.Module):
    def __init__(self, feat_dim=256, stride=2, padding_size=31, fwhm=32, dilation=1, use_checkpoint=False, detach=False, align_corners=True,
                sampler_mode='avgpsp_semantic_edge', 
                padding_mode='reflect', # better than replicate
                # padding_mode='replicate', 
                psp_ratio=4, 
                log_dir=None):
    # def __init__(self,feat_dim=256, stride=2, padding_size=31, fwhm=32, dilation=1, detach=False, use_psp=False, use_avg=True, log_dir=None):
        super(SaliencySampler, self).__init__()
        self.feat_dim = feat_dim
        self.stride = stride
        self.detach = detach
        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.fwhm = fwhm
        self.eps = 1e-16
        psp_bins = [1,2,3,7] if align_corners else [1,2,3,6]
        if 'semantic_edge' in sampler_mode:
            use_psp = ('psp' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            use_softpool = ('softpool' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            if use_avg: 
                assert not use_softpool
                POOL = nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride)
            elif use_softpool:
                POOL = SoftPool2d(kernel_size = 2 * stride + 1, stride=stride, padding=stride)
            else:
                POOL = nn.Identity()
            print(f"*** semantic_edge | use_psp:{use_psp} | use_avg:{use_avg} | use_softpool:{use_softpool} ")
            # _in_channels = int(self.feat_dim * ((1 + 4 / psp_ratio) if use_psp else 1))
            _in_channels = (feat_dim + feat_dim // psp_ratio * 4) if use_psp else feat_dim
            self.saliency_conv = nn.Sequential(
                # nn.AvgPool2d(kernel_size= 5, stride=stride, padding=2),
                # nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                # SoftPool2d(kernel_size = 2 * stride - 1, stride=stride),
                POOL,
                PSP(feat_dim, feat_dim // psp_ratio, bins=psp_bins) if use_psp else nn.Identity(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(_in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(_in_channels, 1, kernel_size=1, padding=0, stride=1)
                # nn.SyncBatchNorm(32),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            )
        elif 'cbam' in sampler_mode:
            use_conv3 = ('3' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            _in_channels = self.feat_dim
            self.saliency_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                ChannelAttention(in_planes=self.feat_dim),
                SpatialAttention(kernel_size=7),
                # LaplacianConv(channels=self.feat_dim, stride=1, padding=1, requires_grad=False), 
                # LHPFConv(channels=self.feat_dim, kernel_size=3, stride=1, padding=1), 
                # LHPFConv(channels=self.feat_dim, kernel_size=5, stride=1, padding=2), 
                # nn.SyncBatchNorm(self.feat_dim), 
                # nn.ReLU(inplace=True),
                nn.Conv2d(_in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(_in_channels, 1, kernel_size=1, padding=0, stride=1)
            )
        elif 'lap' in sampler_mode:
            self.saliency_conv = nn.Sequential(
                LaplacianConv(channels=self.feat_dim, stride=1, padding=1, requires_grad=False), 
                # LHPFConv(channels=self.feat_dim, kernel_size=3, stride=1, padding=1), 
                # LHPFConv(channels=self.feat_dim, kernel_size=5, stride=1, padding=2), 
                nn.SyncBatchNorm(self.feat_dim), 
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feat_dim, 1, kernel_size=3, padding=1, stride=1),
            )
        elif 'lhpf' in sampler_mode: # learnable high pass filter
            use_psp = ('psp' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            use_softpool = ('softpool' in sampler_mode)
            print(f"*** lhpf | use_psp:{use_psp} | use_avg:{use_avg} | use_softpool:{use_softpool} ")
            _in_channels = (feat_dim + feat_dim // psp_ratio * 4) if use_psp else feat_dim
            if use_avg: 
                assert not use_softpool
                POOL = nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride)
            elif use_softpool:
                POOL = SoftPool2d(kernel_size = 2 * stride + 1, stride=stride, padding=stride)
            else:
                POOL = nn.Identity()
            self.saliency_conv = nn.Sequential(
                POOL,
                LHPFConv3(channels=self.feat_dim, stride=1, padding=1, residual=True), 
                # LHPFConv(channels=self.feat_dim, kernel_size=5, stride=1, padding=2), 
                # nn.SyncBatchNorm(self.feat_dim), 
                # nn.ReLU(inplace=True),
                # nn.Conv2d(self.feat_dim, 1, kernel_size=3, padding=1, stride=1),
                PSP(feat_dim, feat_dim // psp_ratio, bins=psp_bins) if use_psp else nn.Identity(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(_in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(_in_channels, 1, kernel_size=1, padding=0, stride=1)
            )
            # self.saliency_conv = nn.Sequential(
            #     InformationEntropy(),
            #     nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1),
            # )
            # self.saliency_conv = nn.Sequential(
            #     MultualInformation(self.feat_dim, mid_channels=self.feat_dim // 16, neighbor_size=3),
            #     nn.Conv2d(9, 1, kernel_size=3, padding=1, stride=1),
            # )
        elif 'cossim' in sampler_mode:
            neighbor_size = 5
            self.saliency_conv = nn.Sequential(
                # LocalCosSim(mid_channels=0, neighbor_size=3), 
                LocalCosSim(in_channels=self.feat_dim, mid_channels=self.feat_dim // 16, neighbor_size=neighbor_size), 
                # LocalCosSim(in_channels=self.feat_dim, mid_channels=32, neighbor_size=3), 
                nn.SyncBatchNorm(neighbor_size ** 2), 
                nn.Conv2d(neighbor_size ** 2, 1, kernel_size=3, padding=1, stride=1),
            )
        elif 'conv' in sampler_mode:
            use_avg = ('avg' in sampler_mode)
            use_conv3 =  ('3' in sampler_mode)
            self.saliency_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                nn.Conv2d(self.feat_dim, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(self.feat_dim, 1, kernel_size=1, padding=0, stride=1)
            )
        elif 'none' in sampler_mode:
            self.saliency_conv = None
            pass
        else:
            raise NotImplementedError
        # nn.init.kaiming_normal_(self.saliency_conv.weight.data)
        # nn.init.kaiming_normal_(self.saliency_conv[1].weight.data)
        # self.bn = nn.BatchNorm2d(self.feat_dim)
        if 'aliasing' in sampler_mode:
            self.aliasing_saliency_conv = nn.Sequential(
                AliasingMapGener(win_size=5, power=2, channel_reduction=16, g_sigma=2, lp_size=4, mask_mode='rect', return_mode='high_ratio'),
                nn.Conv2d(self.feat_dim // 16, 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1)
            )
        else:
            self.aliasing_saliency_conv = None
        
        self.padding_size = padding_size
        
        # init gaussian filter
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm = fwhm))
        self.gaussian_weights = gaussian_weights
        # print("gaussian_weights", gaussian_weights)
        self.filter = nn.Conv2d(1, 1, kernel_size = (2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights
        self.filter.weight.requires_grad = False

        # self.filter = LearnableGaussian(self.padding_size, fwhm, requires_grad=True)

        # coordinate
        self.coord_grid = {} # speedup for different size
        # self.coord_grid = None
        self.reverse_sampler = None

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.writer = SummaryWriter(self.log_dir)

    def gen_coord_grid(self, h, w, device=None):
        """
        return: [2, h, w]
        """
        coord_grid = self.coord_grid.get(f'{h}x{w}', None)
        if coord_grid == None:
        # if self.coord_grid == None \
            # or self.coord_grid.shape[-2] != (h + 2 * self.padding_size) \
                # or self.coord_grid.shape[-1] != (w + 2 * self.padding_size):
            print(f'create grids {h}x{w}...')
            if self.align_corners:
                x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / (w - 1.0)
                y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / (h - 1.0)
            else:
                x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / w + 0.5 / w
                y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / h + 0.5 / h
            coord_grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0)
            # print(coord_grid.unique() * w)
            # coord_grid.requires_grad = False
            if device != None: coord_grid = coord_grid
            self.coord_grid[f'{h}x{w}'] = coord_grid
            # self.coord_grid = coord_grid
            # print(self.coord_grid)
        else:
            pass
        # print(coord_grid.requires_grad)
        return coord_grid.to(device).detach()
        # return self.coord_grid

    def gen_coord_grid_without_pad(self, h, w, device=None):
        coord_grid_without_pad = self.gen_coord_grid(h, w, device)
        return coord_grid_without_pad[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]

    def get_offset(self, grid, cell_encode=True):
        """
        reture abusolute offsets
        """
        b, _, h, w = grid.shape
        # P = torch.zeros(1, 2, h, w)
        # P[0,:,:,:] = self.gen_coord_grid_without_pad(h, w, device=grid.device) * 2 - 1.0
        P = self.gen_coord_grid_without_pad(h, w, device=grid.device) * 2 - 1.0
        # print(self.coord_grid.requires_grad)
        #从[1, 2, 91, 91]扩展为[batch_size, 2, 91, 91]
        # P = P.expand(b, 2, h, w).to(grid.device)
        P = P[None, :, :, :].expand(b, 2, h, w).to(grid.device)
        offset = grid - P
        if cell_encode:
            # offset[:, 0] *= w
            # offset[:, 1] *= h
            cell = torch.ones_like(offset).to(offset.device)
            cell[:, 0] = 2 / w
            cell[:, 1] = 2 / h
            return torch.cat([offset, cell], dim=1)
        else:
            return offset
        
    def create_grid(self, x, h, w):
        """
        x: saliency in [b, 1, h, w]
        """
        #x : 相当于论文中等式2和等式3的S(x',y')* k((x,y),(x',y')),是权重
        #P : 相当于论文中等式2和等式3的[x',y']
        # P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)
        # P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cpu(),requires_grad=False)
        # x = nn.ReplicationPad2d(self.padding_size)(x) #避免等式2和3偏向于图像中心的采样偏好
        # x = nn.ReflectionPad2d(self.padding_size)(x)
        x = F.pad(x, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size
        # P = torch.zeros(1, 2, padded_h, padded_w).to(x.device)
        # P[0,:,:,:] = self.gen_coord_grid(h, w, device=x.device)
        P = self.gen_coord_grid(h, w, device=x.device)[None, ]
        # print(self.coord_grid.requires_grad)
        #从[1, 2, 91, 91]扩展为[batch_size, 2, 91, 91]
        P = P.expand(x.size(0), 2, padded_h, padded_w).to(x.device) 
        # print("P", P.type())
        # print("P", P.shape)
        # print("P size is : ", P.size()) #torch.Size([5, 2, 91, 91])

        x_cat = torch.cat((x, x), 1) #[batch_size, 2, 91, 91]
        # print("x_cat", x_cat.type())
        # print("x_cat size is : ", x_cat.size()) #torch.Size([5, 2, 91, 91])
        #得到的是论文中等式2的分母
        # print("x size is : ", x.size()) #torch.Size([5, 2, 91, 91])
        p_filter = self.filter(x).float() #输入[batch_size, 1, 91, 91]，输出[batch_size, 1, 31, 31]
        # print("x", x.type())
        # print("p_filter", p_filter.type())
        # print("self.filter.weight", self.filter.weight.type())
        # print(self.filter.weight.requires_grad)
        # print("p_filter is : ", p_filter)
        # print("p_filter size is : ", p_filter.size()) #torch.Size([5, 1, 31, 31])


        #得到的是论文中等式2和等式3的分子
        x_mul = torch.mul(P, x_cat).view(-1, 1, padded_h, padded_w) #[batch_size*2, 1, 91, 91]
        # print("x_mul size is : ", x_mul.size()) #torch.Size([10, 1, 91, 91])
        #filter()输入[batch_size*2, 1, 91, 91], 输出[batch_size*2, 1, 31, 31]
        #然后重置为[batch_size, 2, 31, 31]
        all_filter = self.filter(x_mul).view(-1, 2, h, w)
        # print("all_filter size is : ", all_filter.size()) #torch.Size([5, 2, 31, 31])

        # x_filter是u(x,y)的分子,y_filter是v(x,y)的分子
        x_filter = all_filter[:,0,:,:].contiguous().view(-1, 1, h, w) #[batch_size, 1, 31, 31]
        y_filter = all_filter[:,1,:,:].contiguous().view(-1, 1, h, w) #[batch_size, 1, 31, 31]
        # print("y_filter size is : ", y_filter.size()) #torch.Size([5, 1, 31, 31])

        #值的范围是[0,1]
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter + self.eps) #v(x,y)
        # print("y_filter is : ", y_filter)
        # print("y_filter max is : ", y_filter.max()) #tensor(1.0341, grad_fn=<MaxBackward1>)
        # print("y_filter min is : ", y_filter.min()) #tensor(-0.0268, grad_fn=<MinBackward1>)

        #将值的范围从[0,1]改为[-1,1]
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        # print("ygrids max is : ", ygrids.max()) #tensor(1.0200, grad_fn=<MaxBackward1>)
        # print("ygrids min is : ", ygrids.min()) #tensor(-1.0502, grad_fn=<MinBackward1>)
        clamp_value_x = 1 if self.align_corners else (1 - 0.5 / w * 2) 
        clamp_value_y = 1 if self.align_corners else (1 - 0.5 / h * 2) 
        # print(clamp_value_x, clamp_value_y)
        xgrids = torch.clamp(xgrids,min=-clamp_value_x, max=clamp_value_x) #将里面的值的范围控制在[-1,1]
        ygrids = torch.clamp(ygrids,min=-clamp_value_y, max=clamp_value_y)


        xgrids = xgrids.view(-1, 1, h, w)
        ygrids = ygrids.view(-1, 1, h, w)

        grid = torch.cat((xgrids, ygrids), 1) #[batch_size, 2, 31, 31]
        # print('grid', grid.type())
        # grid = grid.type_as(x) # 转换为同数据格式 half float
        # print("grid", grid.shape)
        # print("grid", grid[:, 0, :, :])
        # print("grid", grid[:, 1, :, :])

        # nh, nw = h // self.stride + h % self.stride, w // self.stride + w % self.stride
        # grid = nn.Upsample(size=(nh, nw), mode='bilinear')(grid) #上采样为[batch_size, 2, 224, 224]

        # TODO: check here
        # print(x.max())
        # print(grid[:, :, [0, -1], [0, -1]])
        # grid[:, :, 0, 0] = P[:, :, self.padding_size, self.padding_size] * 2 - 1.0
        # grid[:, :, 0, -1] = P[:, :, self.padding_size, -self.padding_size-1] * 2 - 1.0
        # grid[:, :, -1, 0] = P[:, :, -self.padding_size-1, self.padding_size] * 2 - 1.0
        # grid[:, :, -1, -1] = P[:, :, -self.padding_size-1, -self.padding_size-1] * 2 - 1.0

        grid[:, 1, 0, :] = P[:, 1, self.padding_size, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, 1, -1, :] = P[:, 1, -self.padding_size-1, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, 0, :, 0] = P[:, 0, self.padding_size:-self.padding_size, self.padding_size] * 2 - 1.0
        grid[:, 0, :, -1] = P[:, 0, self.padding_size:-self.padding_size, -self.padding_size-1] * 2 - 1.0
        # print((grid[:, 1, -1, :] * 0.5 + 0.5 - 0.5 / w) * w)
        # grid[:, :, 0, :] = P[:, :, self.padding_size, self.padding_size:-self.padding_size] * 2 - 1.0
        # grid[:, :, -1, :] = P[:, :, -self.padding_size-1, self.padding_size:-self.padding_size] * 2 - 1.0
        # grid[:, :, :, 0] = P[:, :, self.padding_size:-self.padding_size, self.padding_size] * 2 - 1.0
        # grid[:, :, :, -1] = P[:, :, self.padding_size:-self.padding_size, -self.padding_size-1] * 2 - 1.0

        # print(grid.isnan().long().unique())
        # print(type(grid), type(P))
        # print(grid.dtype, P.dtype)
        # grid[grid.isnan()] = P[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size][grid.isnan()] * 2 - 1.0
        # print(grid[:, :, 0, :])
        # print(grid[:, :, -1, :])
        # print(grid[:, :, 1, 0])
        # print(grid[:, :, 1, 1])

        # grid = F.interpolate(grid, size=(nh, nw), mode='bilinear', align_corners=True) #上采样为[batch_size, 2, 224, 224]

        # print("grid", grid.shape)
        # print("grid", grid[:, 0, :, :])
        # print("grid", grid[:, 1, :, :])
        # print(P[:, :, -self.padding_size-1, -self.padding_size-1] * 2 - 1.0)

        # grid = torch.transpose(grid,1,2) #[batch_size, 224, 2, 224]
        # print("grid size is : ", grid.size()) #torch.Size([5, 224, 2, 224])
        # grid = torch.transpose(grid,2,3) #[batch_size, 224, 224, 2]
        # print("grid size is : ", grid.size()) #torch.Size([5, 224, 224, 2])
        # grid = grid.permute(0, 2, 3, 1)
        # print('grid', grid.type())
        return grid # in x, y

    def create_grid_inv(self, grid, target_h, target_w):
        """
        grid in (N,2,H,W)
        """
        # say original image size (U,V), downsampled size (X,Y)
        # with grid of size (N,2,H,W)
        # grid_reorder = grid.permute(0,3,1,2)
        # with grid of size (2,N,H,W)
        grid_reorder = grid.permute(1, 0, 2, 3)
        # we know for each x,y in grid, its corresponding u,v saved in [N,0,x,y] and [N,1,x,y], which are in range [-1,1]
        # N is batch size
        # grid_inv = torch.ones(grid_reorder.shape[0],2,segSize[0],segSize[1])*-1

        grid_inv = torch.autograd.Variable(torch.zeros((2, grid_reorder.shape[1], target_h, target_w), device=grid_reorder.device))
        grid_inv[:] = float('nan')

        # size (2,N,U,V) -> (2,N,H,W)

        u_cor = (((grid_reorder[0,:,:,:]+1)/2)*(target_w-1)).int().long().reshape(grid_reorder.shape[1], -1)
        v_cor = (((grid_reorder[1,:,:,:]+1)/2)*(target_h-1)).int().long().reshape(grid_reorder.shape[1], -1)
        x_cor = torch.arange(0,grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        y_cor = torch.arange(0,grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(x_cor)
        grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(y_cor)

        # rearange to [-1,1]
        # grid_inv[:,0,:,:] = grid_inv[:,0,:,:]/grid_reorder.shape[3]*2-1
        # grid_inv[:,1,:,:] = grid_inv[:,1,:,:]/grid_reorder.shape[2]*2-1
        grid_inv[0] = grid_inv[0]/grid_reorder.shape[3]*2-1
        grid_inv[1] = grid_inv[1]/grid_reorder.shape[2]*2-1

        # grid_inv = torch.tensor(grid_inv)
        # (N,2,H,W) -> (N,H,W,2)
        # grid_inv = grid_inv.permute(0,2,3,1)
        # (2,N,H,W) -> (N,H,W,2)
        grid_inv = grid_inv.permute(1, 0, 2, 3)
        # (2,N,H,W) -> (N,H,W,2)
        # grid_inv = grid_inv.permute(1,2,3,0)
        return grid_inv
    

    def reverse(self, sampled_feat, grid, h, w, mode='barycentric'):
        if mode =='IDW':
            return self.reverse_IDW(sampled_feat, grid, h, w)
        return self.reverse_barycentric(sampled_feat, grid, h, w)

    def reverse_IDW(self, sampled_feat, grid, h, w):
        # 确保grid的形状是 [B, H_in, W_in, 2]
        if grid.size(1) == 2 and grid.size(3) != 2:
            grid = grid.permute(0, 2, 3, 1)
        
        # 调用优化后的分块上采样函数
        tile_size = 64
        search_margin = int(round(h / grid.size(1))) * 2
        # print(f"tile_size: {tile_size}, search_margin: {search_margin}")
        reversed_feats = optimized_tiled_non_uniform_upsample(
            sampled_feat, 
            grid, 
            target_h=h, 
            target_w=w,
            k=3, 
            power=2,
            tile_size=tile_size, # 可调参数
            search_margin=search_margin  # 可调参数
        )

        if self.log_dir is not None:
            vis_reversed_feats = reversed_feats.detach()
            self.writer.add_images(f"reversed_feats{h}x{w}", ((vis_reversed_feats.mean(dim=1, keepdim=True) - vis_reversed_feats.min()) / (vis_reversed_feats.max() - vis_reversed_feats.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
        
        return reversed_feats

    def reverse_barycentric(self, sampled_feat, grid, h, w):
        if self.reverse_sampler == None or self.reverse_sampler.h != h or self.reverse_sampler.w != w:
            self.reverse_sampler = Interp2D(h, w)
        else:
            pass
        # for grid in b, 2, h, w
        if grid.size(1) == 2 and grid.size(3) != 2:
            grid = grid.permute(0, 2, 3, 1)
        grid = grid.clone().contiguous()
        if self.align_corners:
            grid[:, :, :, 0] = 0.5 * (grid[:, :, :, 0] + 1) * (w - 1)
            grid[:, :, :, 1] = 0.5 * (grid[:, :, :, 1] + 1) * (h - 1)
            # print(grid.long().unique())   
        else:
            grid[:, :, :, 0] = 0.5 * (grid[:, :, :, 0] + 1) - 0.5 / w
            grid[:, :, :, 1] = 0.5 * (grid[:, :, :, 1] + 1) - 0.5 / h
            grid[:, :, :, 0] = grid[:, :, :, 0] * w
            grid[:, :, :, 1] = grid[:, :, :, 1] * h
            grid[:, :, 0, 0] = 0
            grid[:, :, -1, 0] = w - 1
            grid[:, 0, :, 1] = 0
            grid[:, -1, :, 1] = h - 1
            # print(grid.long().unique())   
        # print(grid.shape)
        # print("grid", grid)
        b, _, sf_h, sf_w = sampled_feat.shape

        # grid = grid.reshape(b, sf_h * sf_w, 2)
        grid = grid.view(b, sf_h * sf_w, 2)
        grid = grid[:, :, [1, 0]]
        sampled_feat = sampled_feat.permute(0, 2, 3, 1).view(b, sf_h * sf_w, -1)

        reversed_feats = []
        for i in range(b):
            '''
            New 2d Interpolation in Pytorch
            Reference to scipy.griddata
            Args:
                h, w:  height,width of input
                points: points to interpote shape: [num, 2]
                values:  values of points shape:[num, valuedim]
            return:
            2D interpolate result, shape: [valuedim, h, w]
            '''
            reversed_feat = self.reverse_sampler(grid[i].detach(), sampled_feat[i])
            # reversed_feat = reversed_feat.permute(1, 2, 0)
            reversed_feats.append(reversed_feat)
        reversed_feats = torch.stack(reversed_feats, dim=0)
        if self.log_dir is not None:
            vis_reversed_feats = reversed_feats.detach()
            # vis_reversed_feats = reversed_feats.max(dim=1, keepdim=True)[0].detach()
            # vis_reversed_feats = reversed_feats.mean(dim=1, keepdim=True).detach()
            # vis_reversed_feats = reversed_feats[:, 0:3].detach()
            self.writer.add_images(f"reversed_feats{h}x{w}", ((vis_reversed_feats.mean(dim=1, keepdim=True) - vis_reversed_feats.min()) / (vis_reversed_feats.max() - vis_reversed_feats.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
        return reversed_feats

    def forward(self, feature, sample_with_grid=True):
        # _, c, h, w = feature.shape
        # nh, nw = h // self.stride, w // self.stride
        # saliency = self.saliency_conv(F.relu(feature, inplace=False)) #得到saliency map,channels=1,即[batch_size, c, h, w]
        # saliency = self.saliency_conv(feature) #得到saliency map,channels=1,即[batch_size, c, h, w]
        if self.saliency_conv is None:
            b, _, h, w = feature.shape
            feature_sampled = feature
            saliency = None
            grid = self.gen_coord_grid_without_pad(h, w, device=feature.device)
            grid = grid[None, ].repeat(b, 1, 1, 1) * 2 - 1
            return feature, feature_sampled, saliency, grid
            
        if self.use_checkpoint:
            saliency = checkpoint(self.saliency_conv, feature) #得到saliency map,channels=1,即[batch_size, c, h, w]
            if self.aliasing_saliency_conv is not None:
                saliency += self.aliasing_saliency_conv(feature)
        else:
            saliency = self.saliency_conv(feature)
            if self.aliasing_saliency_conv is not None:
                saliency += self.aliasing_saliency_conv(feature)
                # saliency += self.aliasing_saliency_conv(GetAliasingMap(feature, win_size=5, power=2, channel_reduction=16, g_sigma=2, lp_size=4, mask_mode='rect', return_mode='high_ratio'))
        _, c, h, w = saliency.shape
        # saliency = nn.Upsample(size=(nh, nw), mode='bilinear')(saliency) #上采样为[batch_size, 1, 31, 31]
        # saliency = saliency.view(-1, nh * nw) #重置大小为[batch_size, 31*31]
        # saliency = F.softmax(saliency) #得到每个像素的权重
        # saliency = saliency.view(-1, 1, nh, nw) #再重置为[batch_size, 1, 31, 31]
        # saliency *= 0
        # saliency  = (saliency - saliency.mean()).clamp(min=-1, max=1)
        # saliency  = (saliency - saliency.mean()).clamp(min=-10, max=10)
        # saliency = saliency.clamp(min=-10, max=10)
        saliency = saliency.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w) * h * w
        # saliency = F.interpolate(saliency, (h, w), mode='bilinear', align_corners=True)
        # padded_saliency = nn.ReflectionPad2d(self.padding_size)(saliency) #避免等式2和3偏向于图像中心的采样偏好
        # padded_saliency = nn.ReplicationPad2d(self.padding_size)(saliency) #避免等式2和3偏向于图像中心的采样偏好
        # print("padded_saliency", padded_saliency)
        # grid = self.create_grid(padded_saliency, h, w) # b, h, w, 2
        grid = self.create_grid(saliency, h, w) # b, h, w, 2
        # h, w = h // self.stride + h % self.stride, w // self.stride + w % self.stride
        # grid = F.interpolate(grid.permute(0, 3, 1, 2), (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        # grid = F.interpolate(grid, (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1) # b, h, w, 2
        # grid = grid.permute(0, 2, 3, 1) # b, h, w, 2
        # print(self.saliency_conv[1].weight[0, 0, 0])

        if sample_with_grid:
            # grid = grid.type_as(feature)
            if self.detach:
                feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
            else:
                # print(feature, grid)
                feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        else:
            feature_sampled = None
        """
        reversed_feat = self.reverse(feature_sampled, grid, h, w)
        print((reversed_feat - feature).abs().mean())
        print(saliency.std())
        #"""
        # print(feature.shape, feature_sampled.shape, saliency.shape, grid.shape)
        # print(F.interpolate(grid, (3, 3), mode='bilinear', align_corners=True))
        if False:
            # _feature = feature.detach().cpu()
            _feature = feature.detach()
            # _feature = _feature.mean(dim=1, keepdim=True)
            _feature = (_feature - _feature.min()) / (_feature.max() - _feature.min()) * 2047 + 1
            # _feature = _feature.sigmoid() * 1023 + 1
            feature2x = F.interpolate(_feature, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
            # feature2x = F.interpolate(_feature, size=(h // 2 + 1, w // 2 + 1), mode='nearest')
            feature2x = F.interpolate(feature2x, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            feature4x = F.interpolate(_feature, size=(h // 4 + 1, w // 4 + 1), mode='bilinear', align_corners=True)
            # feature4x = F.interpolate(_feature, size=(h // 4 + 1, w // 4 + 1), mode='nearest')
            # feature4x = F.interpolate(feature4x, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
            feature4x = F.interpolate(feature4x, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            feature8x = F.interpolate(_feature, size=(h // 8 + 1, w // 8 + 1), mode='bilinear', align_corners=True)
            # feature8x = F.interpolate(_feature, size=(h // 8 + 1, w // 8 + 1), mode='nearest')
            # feature8x = F.interpolate(feature8x, size=(h // 4 + 1, w // 4 + 1), mode='bilinear', align_corners=True)
            # feature8x = F.interpolate(feature8x, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
            feature8x = F.interpolate(feature8x, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            _feature_sampled = feature_sampled.detach()
            # _feature_sampled = feature_sampled.detach().cpu()
            # _feature_sampled = _feature_sampled.mean(dim=1, keepdim=True) 
            _feature_sampled = (_feature_sampled - _feature_sampled.min()) / (_feature_sampled.max() - _feature_sampled.min()) * 2047 + 1
            # _feature_sampled = (_feature_sampled - feature.min()) / (feature.max() - feature.min()) * 2047 + 1
            # _feature_sampled = _feature_sampled.sigmoid() * 1023 + 1
            feature2x_sampled = F.interpolate(_feature_sampled, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
            # feature2x_sampled = F.interpolate(_feature_sampled, size=(h // 2 + 1, w // 2 + 1), mode='nearest')
            feature2x_sampled = F.interpolate(feature2x_sampled, size=(h, w), mode='bilinear', align_corners=True)
            feature2x_sampled_reversed = self.reverse(feature2x_sampled, grid, h * 2 - 1, w * 2 - 1)
            # feature2x_sampled_reversed = self.reverse(feature2x_sampled, grid, h * 4 - 3, w * 4 - 3)
            feature2x_sampled_reversed = F.interpolate(feature2x_sampled_reversed, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            feature4x_sampled = F.interpolate(_feature_sampled, size=(h // 4 + 1, w // 4 + 1), mode='bilinear', align_corners=True)
            # feature4x_sampled = F.interpolate(_feature_sampled, size=(h // 4 + 1, w // 4 + 1), mode='nearest')
            feature4x_sampled = F.interpolate(feature4x_sampled, size=(h, w), mode='bilinear', align_corners=True)
            feature4x_sampled_reversed = self.reverse(feature4x_sampled, grid, h * 2 - 1, w * 2 - 1)
            # feature4x_sampled_reversed = self.reverse(feature4x_sampled, grid, h * 4 - 3, w * 4 - 3)
            feature4x_sampled_reversed = F.interpolate(feature4x_sampled_reversed, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            feature8x_sampled = F.interpolate(_feature_sampled, size=(h // 8 + 1, w // 8 + 1), mode='bilinear', align_corners=True)
            # feature8x_sampled = F.interpolate(_feature_sampled, size=(h // 8 + 1, w // 8 + 1), mode='nearest')
            feature8x_sampled = F.interpolate(feature8x_sampled, size=(h, w), mode='bilinear', align_corners=True)
            feature8x_sampled_reversed = self.reverse(feature8x_sampled, grid, h * 2 - 1, w * 2 - 1)
            # feature8x_sampled_reversed = self.reverse(feature8x_sampled, grid, h * 4 - 3, w * 4 - 3)
            feature8x_sampled_reversed = F.interpolate(feature8x_sampled_reversed, size=(h, w), mode='bilinear', align_corners=True)[:, :, :-1, :-1]

            uniform_2x_dct.save(torch_patchwise_dct(feature2x.cpu()))
            uniform_4x_dct.save(torch_patchwise_dct(feature4x.cpu()))
            uniform_8x_dct.save(torch_patchwise_dct(feature8x.cpu()))

            adaptive_2x_dct.save(torch_patchwise_dct(feature2x_sampled_reversed.cpu()))
            adaptive_4x_dct.save(torch_patchwise_dct(feature4x_sampled_reversed.cpu()))
            adaptive_8x_dct.save(torch_patchwise_dct(feature8x_sampled_reversed.cpu()))

            original_dct.save(torch_patchwise_dct(_feature.cpu()))


        if self.log_dir is not None:
            grid_vis = grid.detach()
            # grid_vis = grid.clone().permute(0, 3, 1, 2)
            # print(self.coord_grid.shape)
            # grid_vis -= self.coord_grid[None, ][:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            # grid_vis = self.get_offset(grid_vis)
            # grid_vis[:, 0] *= 0.5 * grid_vis.size(-1) # x, w
            # grid_vis[:, 1] *= 0.5 * grid_vis.size(-2) # y, h
            print('pred_grid', grid_vis.abs().max())
            grid_vis = (grid_vis - grid_vis.min()) / (grid_vis.max() - grid_vis.min())
            # print(grid_vis.shape)
            self.writer.add_images(f"gridx{h}x{w}", grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"gridy{h}x{w}", grid_vis[:, 1:2], global_step=None, walltime=None, dataformats='NCHW')
            # self.writer.add_images(f"gridxy{h}x{w}", grid_vis[:, :], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"saliency{h}x{w}", ((saliency - saliency.min()) / (saliency.max() - saliency.min())) * 25, global_step=None, walltime=None, dataformats='NCHW')
            filter_weight = self.filter.weight[0].data[:,:,:]
            # filter_weight = self.filter_weight()[0]
            # print(self.sigma)
            # gaussian_weight = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm = 8))[None, ]
            # self.writer.add_images("filter", (filter_weight - filter_weight.min()) / (filter_weight.max() - filter_weight.min()) * 64., global_step=None, walltime=None, dataformats='CHW')
            self.writer.add_images(f"filter{h}x{w}", filter_weight, global_step=None, walltime=None, dataformats='CHW')
            # self.writer.add_images("gaussian_weight", gaussian_weight, global_step=None, walltime=None, dataformats='CHW')
            # vis_feature = feature.mean(dim=1, keepdim=True)
            vis_feature = feature.detach()
            self.writer.add_images(f"feature{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
            
            if feature_sampled is not None:
                vis_feature = feature_sampled.detach()
                self.writer.add_images(f"feature_sampled{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
                
                vis_feature = F.interpolate(vis_feature, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
                vis_feature = F.interpolate(vis_feature, size=(h, w), mode='bilinear', align_corners=True)
                reversed_feat = self.reverse(vis_feature, grid, h * 4 - 3, w * 4 - 3)
                vis_feature = reversed_feat.detach()
                _, _, h, w = vis_feature.shape
                self.writer.add_images(f"feature_reversed{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
            # self.writer.add_images("feature", (feature.max(dim=1, keepdim=True)[0] - feature.min()) / (feature.max() - feature.min()) * 64., global_step=None, walltime=None, dataformats='NCHW')
        return feature, feature_sampled, saliency, grid
        # return feature, feature_sampled, reversed_feat, saliency, grid
    
    def sample_with_grid(self, feature, grid, h=None, w=None):
        if self.detach:
            feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=True, padding_mode="border") #得到重采样的图像
        else:
            feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True, padding_mode="border") #得到重采样的图像
        return feature_sampled

    def sample_with_saliency(self, feature, saliency, h=None, w=None):
        # saliency = saliency.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w)
        if h == None:
            _, _, h, w = feature.shape
        saliency = F.interpolate(saliency, (h, w), mode='bilinear', align_corners=self.align_corners, padding_mode="border")
        # padded_saliency = nn.ReplicationPad2d(self.padding_size)(saliency) #避免等式2和3偏向于图像中心的采样偏好
        # print("padded_saliency", padded_saliency)
        grid = self.create_grid(saliency, h, w).permute(0, 2, 3, 1)
        feature_sampled = F.grid_sample(feature, grid, mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        """
        reversed_feat = self.reverse(feature_sampled, grid, h, w)
        print((reversed_feat - feature).abs().mean())
        print(saliency.std())
        #"""
        return feature, feature_sampled, saliency, grid
        # return feature_sampled
        # return feature, feature_sampled, reversed_feat, saliency, grid