
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
import mmseg.models.backbones.spatial.qhull as qhull
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
    Learnable High Pass Filter
    """
    def __init__(self, channels=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            # [1/8., 1/8., 1/8.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1).contiguous()
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        identity_kernel = [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ]
        identity_kernel = torch.FloatTensor(identity_kernel).unsqueeze(0).unsqueeze(0)
        self.identity_kernel = nn.Parameter(data=identity_kernel, requires_grad=False)
 
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.identity_kernel - self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3), 
                    padding=self.padding, groups=self.channels, stride=self.stride)
        return x

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

class LaplacianConvWithMean(nn.Module):
    """
    Learnable Low Pass Filter
    """
    def __init__(self, stride=2, padding=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.lap_conv=LaplacianConv(channels=1, stride=stride, padding=0)
 
    def forward(self, x):
        x = x.mean(dim=1)[:, None]
        x = self.pad(x)
        x = self.lap_conv(x)
        return x


def test_makeGaussian():
    # kernel = makeGaussian(size = 5, fwhm = 1)
    # print(kernel)
    x = torch.rand(2, 1, 16, 16)
    m = LearnableGaussian()
    print(m(x))
    print(m.weight)


class MultualInformation(nn.Module):
    def __init__(self, in_channels, mid_channels=8, neighbor_size=3, stride=1, dilation=1, eps=1e-8):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.pad = neighbor_size // 2
        self.neighbor_size = neighbor_size
        self.stride = stride
        self.dilation = dilation
        self.eps = eps
        pass

    def forward(self, x):
        feat = self.compress(x)
        # feat = feat - feat.mean(dim=1)[:, None]
        # feat = feat.pow(2)
        # feat = feat - feat.max(dim=1, keepdim=True)[0]
        px = feat.softmax(dim=1)
        b, c, h, w = px.shape
        py = F.pad(px, pad=[self.pad] * 4, mode='reflect')
        # shape:  B x (C x ksize x ksize) x H // stride x W //stride
        py = F.unfold(py, kernel_size=(self.neighbor_size, self.neighbor_size), stride=self.stride, dilation=self.dilation)
        # py = py.reshape(b, c, self.neighbor_size, self.neighbor_size, h, w)
        # center = self.neighbor_size // 2
        IEX = (- px * px.log()).sum(dim=1)[:, None] # b, 1, h, w
        IEY = F.pad(IEX, pad=[self.pad] * 4, mode='reflect')
        IEY = F.unfold(IEY, kernel_size=(self.neighbor_size, self.neighbor_size), stride=self.stride, dilation=self.dilation) # b, (k, k), h, w
        IEY = IEY.reshape(b, self.neighbor_size ** 2, h, w)
        # PXY = torch.einsum("bciihw,bckkhw->bcckkhw", [px, p]
        py = py.reshape(b, c, 1, self.neighbor_size ** 2 * h * w).permute(0, 3, 1, 2) # b, (k, k, h, w) c, 1
        px = px.repeat(1, self.neighbor_size ** 2, 1, 1)
        px = px.reshape(b, 1, c, self.neighbor_size ** 2 * h * w).permute(0, 3, 1, 2) # b, (k, k, h, w) 1, c
        pxy = py @ px
        pxy = pxy.reshape(b, self.neighbor_size ** 2, h, w, c ** 2)
        IEXY = (- pxy * pxy.log()).sum(dim=-1) # b, (k, k), h, w
        MI = IEX.repeat(1, self.neighbor_size ** 2, 1, 1) + IEY - IEXY
        print(IEX.shape, IEY.shape, IEXY.shape)
        print(IEX[:, :, 0, 0])
        print(IEY[:, :, 0, 0])
        print(IEXY[:, :, 0, 0])
        # print(IEY[:, 0])
        # print(IEXY[:, 4])
        print(MI.max())
        return MI
    
def test_MultualInformation():
    x = torch.rand(1, 128, 4, 4) * 4
    x[:, :, 0, 0] += 10
    m = MultualInformation(128, 16)
    print(m(x).shape)
    
class InformationEntropy(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        # self.pad = nn.ReflectionPad2d(padding)
        # self.lap_conv=LaplacianConv(channels=1, stride=stride, padding=0)
 
    def forward(self, x):
        x_2 = x - x.mean(dim=1)[:, None]
        x_2 = x_2.pow(2)
        x_2 = x_2 - x_2.max(dim=1, keepdim=True)[0]
        x_2 = x_2.softmax(dim=1)
        h = - x_2 * (x_2 + self.eps).log()
        h = h.sum(dim=1)[:, None]
        return h

def test_InformationEntropy():
    x = torch.rand(2, 256, 16, 16)
    m = InformationEntropy()
    print(m(x))

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

class NonUniformRepresentation(nn.Module):
    def __init__(self, feat_dim=256, stride=1, padding_size=5, fwhm=3, dilation=1, use_checkpoint=True, detach=False, align_corners=True,
                sampler_mode='none', 
                padding_mode='reflect', # better than replicate
                upsample_scale=4,
                # padding_mode='replicate', 
                psp_ratio=4, log_dir=None):
    # def __init__(self,feat_dim=256, stride=2, padding_size=31, fwhm=32, dilation=1, detach=False, use_psp=False, use_avg=True, log_dir=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.stride = stride
        self.detach = detach
        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.upsample_scale = upsample_scale
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
            _in_channels = int(self.feat_dim * ((1 + 4 / psp_ratio) if use_psp else 1))
            self.saliency_conv = nn.Sequential(
                # nn.AvgPool2d(kernel_size= 5, stride=stride, padding=2),
                # nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                # SoftPool2d(kernel_size = 2 * stride - 1, stride=stride),
                POOL,
                PSP(feat_dim, feat_dim // psp_ratio) if use_psp else nn.Identity(),
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
            self.saliency_conv = nn.Sequential(
                LHPFConv(channels=self.feat_dim, kernel_size=3, stride=1, padding=1), 
                # LHPFConv(channels=self.feat_dim, kernel_size=5, stride=1, padding=2), 
                nn.SyncBatchNorm(self.feat_dim), 
                nn.ReLU(inplace=True),
                # nn.Conv2d(self.feat_dim, 1, kernel_size=3, padding=1, stride=1),
                PSP(feat_dim, feat_dim // 4) if use_psp else nn.Identity(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(self.feat_dim * (2 if use_psp else 1), 1, kernel_size=3, padding=dilation, dilation=dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(self.feat_dim * (2 if use_psp else 1), 1, kernel_size=1, padding=0, stride=1)
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
            self.saliency_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                nn.Conv2d(self.feat_dim, 1, kernel_size=1, padding=0, stride=1),
            )
        elif 'shuffle' in sampler_mode:
            # use_avg = ('avg' in sampler_mode)
            self.saliency_conv = nn.Sequential(
                # nn.AvgPool2d(kernel_size= 2 * stride + 1, stride=stride, padding=stride) if use_avg else nn.Identity(),
                nn.Conv2d(self.feat_dim, 1 * self.upsample_scale ** 2, kernel_size=1, padding=0, stride=1),
                nn.PixelShuffle(self.upsample_scale)
            )
        elif 'none' in sampler_mode:
            self.saliency_conv = None
            pass
        else:
            raise NotImplementedError
        # nn.init.kaiming_normal_(self.saliency_conv.weight.data)
        # nn.init.kaiming_normal_(self.saliency_conv[1].weight.data)
        # self.bn = nn.BatchNorm2d(self.feat_dim)
        
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
            if device != None: coord_grid = coord_grid.to(device)
            self.coord_grid[f'{h}x{w}'] = coord_grid
            # self.coord_grid = coord_grid
        else:
            pass
        # print(coord_grid.requires_grad)
        return coord_grid.detach()
        # return self.coord_grid

    def gen_coord_grid_without_pad(self, h, w, device=None):
        coord_grid_without_pad = self.gen_coord_grid(h, w, device)
        return coord_grid_without_pad[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]

    def get_offset(self, grid, cell_encode=False):
        """
        reture abusolute offsets
        """
        b, _, h, w = grid.shape
        P = torch.zeros(1, 2, h, w)
        P[0,:,:,:] = self.gen_coord_grid_without_pad(h, w, device=grid.device) * 2 - 1.0
        # print(self.coord_grid.requires_grad)
        #从[1, 2, 91, 91]扩展为[batch_size, 2, 91, 91]
        P = P.expand(b, 2, h, w).to(grid.device)
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
        P = torch.zeros(1, 2, padded_h, padded_w).to(x.device)
        P[0,:,:,:] = self.gen_coord_grid(h, w, device=x.device)
        # print(self.coord_grid.requires_grad)
        #从[1, 2, 91, 91]扩展为[batch_size, 2, 91, 91]
        P = P.expand(x.size(0), 2, padded_h, padded_w).to(x.device) 
        # print("P", P.shape)
        # print("P", P)
        # print("P size is : ", P.size()) #torch.Size([5, 2, 91, 91])

        x_cat = torch.cat((x, x), 1) #[batch_size, 2, 91, 91]
        # print("x_cat size is : ", x_cat.size()) #torch.Size([5, 2, 91, 91])
        #得到的是论文中等式2的分母
        # print("x size is : ", x.size()) #torch.Size([5, 2, 91, 91])
        p_filter = self.filter(x) #输入[batch_size, 1, 91, 91]，输出[batch_size, 1, 31, 31]
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
        x_filter = x_filter/p_filter #u(x,y)
        y_filter = y_filter/p_filter #v(x,y)
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

        if self.log_dir is not None and False:
            grid_vis = grid.detach()
            # grid_vis = grid.clone().permute(0, 3, 1, 2)
            # print(self.coord_grid.shape)
            # grid_vis -= self.coord_grid[None, ][:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            # grid_vis = self.get_offset(grid_vis)
            # grid_vis[:, 0] *= 0.5 * grid_vis.size(-1) # x, w
            # grid_vis[:, 1] *= 0.5 * grid_vis.size(-2) # y, h
            print('pred_grid', grid_vis.abs().max())
            grid_vis = self.get_offset(grid_vis, cell_encode=False)
            grid_vis = (grid_vis - grid_vis.min()) / (grid_vis.max() - grid_vis.min())
            # print(grid_vis.shape)
            self.writer.add_images(f"gridx{h}x{w}", grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"gridy{h}x{w}", grid_vis[:, 1:2], global_step=None, walltime=None, dataformats='NCHW')
            # self.writer.add_images(f"gridxy{h}x{w}", grid_vis[:, :], global_step=None, walltime=None, dataformats='NCHW')
            saliency = x.detach()
            self.writer.add_images(f"saliency{h}x{w}", ((saliency - saliency.min()) / (saliency.max() - saliency.min())) * 25, global_step=None, walltime=None, dataformats='NCHW')
        return grid # in x, y

    def reverse(self, sampled_feat, grid, h, w):
        if self.log_dir is not None:
            grid_vis = grid.detach()
            # grid_vis = grid.clone().permute(0, 3, 1, 2)
            # print(self.coord_grid.shape)
            # grid_vis -= self.coord_grid[None, ][:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            # grid_vis = self.get_offset(grid_vis)
            # grid_vis[:, 0] *= 0.5 * grid_vis.size(-1) # x, w
            # grid_vis[:, 1] *= 0.5 * grid_vis.size(-2) # y, h
            print('pred_grid', grid_vis.abs().max())
            grid_vis = self.get_offset(grid_vis, cell_encode=False)
            grid_vis = (grid_vis - grid_vis.min()) / (grid_vis.max() - grid_vis.min())
            # print(grid_vis.shape)
            self.writer.add_images(f"gridx{h}x{w}", grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"gridy{h}x{w}", grid_vis[:, 1:2], global_step=None, walltime=None, dataformats='NCHW')
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
            # self.writer.add_images(f"reversed_feats{h}x{w}", ((vis_reversed_feats.mean(dim=1, keepdim=True) - vis_reversed_feats.min()) / (vis_reversed_feats.max() - vis_reversed_feats.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"reversed_feats{h}x{w}", vis_reversed_feats.max(dim=1, keepdim=True)[1] * 10, global_step=None, walltime=None, dataformats='NCHW')
        return reversed_feats


    def forward(self, feature, only_logits=False):
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
        else:
            saliency = self.saliency_conv(feature)
        _, c, h, w = saliency.shape
        # saliency = nn.Upsample(size=(nh, nw), mode='bilinear')(saliency) #上采样为[batch_size, 1, 31, 31]
        # saliency = saliency.view(-1, nh * nw) #重置大小为[batch_size, 31*31]
        # saliency = F.softmax(saliency) #得到每个像素的权重
        # saliency = saliency.view(-1, 1, nh, nw) #再重置为[batch_size, 1, 31, 31]
        # saliency *= 0
        # saliency  = (saliency - saliency.mean()).clamp(min=-1, max=1)
        # saliency  = (saliency - saliency.mean()).clamp(min=-10, max=10)
        # saliency = saliency.clamp(min=-10, max=10)
        if only_logits:
            return saliency
        saliency = saliency.view(-1, h * w).softmax(dim=-1).view(-1, 1, h, w)
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

        # if sample_with_grid:
        #     # grid = grid.type_as(feature)
        #     if self.detach:
        #         feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1).detach(), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        #     else:
        #         feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        # else:
        #     feature_sampled = None
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
            # print('pred_grid', grid_vis.abs().max())
            grid_vis = self.get_offset(grid_vis)
            grid_vis = (grid_vis - grid_vis.min()) / (grid_vis.max() - grid_vis.min())
            # print(grid_vis.shape)
            self.writer.add_images(f"gridx{h}x{w}", grid_vis[:, 0:1], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"gridy{h}x{w}", grid_vis[:, 1:2], global_step=None, walltime=None, dataformats='NCHW')
            # self.writer.add_images(f"gridxy{h}x{w}", grid_vis[:, :], global_step=None, walltime=None, dataformats='NCHW')
            self.writer.add_images(f"saliency{h}x{w}", ((saliency - saliency.min()) / (saliency.max() - saliency.min())), global_step=None, walltime=None, dataformats='NCHW')
            filter_weight = self.filter.weight[0].data[:,:,:]
            # filter_weight = self.filter_weight()[0]
            # print(self.sigma)
            # gaussian_weight = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm = 8))[None, ]
            # self.writer.add_images("filter", (filter_weight - filter_weight.min()) / (filter_weight.max() - filter_weight.min()) * 64., global_step=None, walltime=None, dataformats='CHW')
            self.writer.add_images(f"filter{h}x{w}", filter_weight, global_step=None, walltime=None, dataformats='CHW')
            # self.writer.add_images("gaussian_weight", gaussian_weight, global_step=None, walltime=None, dataformats='CHW')
            # vis_feature = feature.mean(dim=1, keepdim=True)
            # vis_feature = feature.detach()
            # self.writer.add_images(f"feature{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
            
            # if feature_sampled is not None:
            #     vis_feature = feature_sampled.detach()
            #     self.writer.add_images(f"feature_sampled{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
                
            #     vis_feature = F.interpolate(vis_feature, size=(h // 2 + 1, w // 2 + 1), mode='bilinear', align_corners=True)
            #     vis_feature = F.interpolate(vis_feature, size=(h, w), mode='bilinear', align_corners=True)
            #     reversed_feat = self.reverse(vis_feature, grid, h * 4 - 3, w * 4 - 3)
            #     vis_feature = reversed_feat.detach()
            #     _, _, h, w = vis_feature.shape
            #     self.writer.add_images(f"feature_reversed{h}x{w}", ((vis_feature.mean(dim=1, keepdim=True) - vis_feature.min()) / (vis_feature.max() - vis_feature.min())) * 25., global_step=None, walltime=None, dataformats='NCHW')
            # # self.writer.add_images("feature", (feature.max(dim=1, keepdim=True)[0] - feature.min()) / (feature.max() - feature.min()) * 64., global_step=None, walltime=None, dataformats='NCHW')
        return saliency, grid
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

    
if __name__=='__main__':
    # test_2()
    # test_coordinate_generate()
    # test_torch_patchwise_dct()
    # test_SaliencySampler()
    # test_makeGaussian()
    test_softpool()
    # test_makeGaussian()
    # test_InformationEntropy()
    # test_MultualInformation()
    # test_LocalCosSim()
