
# from operator import index
from cgi import test
from turtle import forward
import warnings

import math
import cv2
from torch import nn
from torch.autograd import Function
import torch
from torch.nn.common_types import _size_2_t
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
    def __init__(self, channels=3, stride=2, padding=1, residual=False):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.residual = residual
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
        if self.residual: identity_kernel = identity_kernel * 2 
        self.identity_kernel = nn.Parameter(data=identity_kernel, requires_grad=False)
        # print(self.identity_kernel)
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # if self.residual:
            # residual = x
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

class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)
    
def generate_circle_mask(radius, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask[(x - center)**2 + (y - center)**2 <= radius**2] = 1
    return mask

def generate_gaussian_kernel(kernel_size=5, sigma=1):
    """
    Generate a Gaussian kernel with the given size and standard deviation.
    
    Parameters:
        kernel_size (int): Size of the kernel (should be an odd number).
        sigma (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        np.ndarray: Gaussian kernel of shape (kernel_size, kernel_size).
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number.")
    
    # Generate a 2D grid of coordinates centered at the kernel
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(x, y)
    
    # Compute the Gaussian kernel using the 2D coordinates and the given standard deviation
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    
    # Normalize the kernel to make sure it sums to 1
    kernel /= kernel.sum()
    
    return kernel

def GetAliasingMap(tensor, win_size=5, power=2, channel_reduction=4, g_sigma=2, lp_size=4, mask_mode='rect', return_mode='high_ratio'):
    """
    lp_size = 4 -> 1/4, half of highest frequency
    """
    b, c, h, w = tensor.shape
    tensor = tensor.reshape(b, c // channel_reduction, channel_reduction, h, w)
    tensor = tensor.mean(-3)
    b, c, h, w = tensor.shape
    tensor = F.unfold(F.pad(tensor, pad=[win_size//2] * 4, mode='replicate'),
                    kernel_size=win_size, dilation=1, padding=0, stride=1)
    tensor = tensor.reshape(b, -1, h, w)
    tensor = tensor.permute(0, 2, 3, 1) # b, h, w, cxkxk
    tensor = tensor.reshape(b, h * w * c, win_size, win_size) # b, h * w * c, k, k
    G = generate_gaussian_kernel(win_size, sigma=g_sigma)
    G = torch.Tensor(G).reshape(1, 1, win_size, win_size)
    tensor = G * tensor
    # 将输入tensor转换为频域
    freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
    # 将低频部分移动到频域中心
    shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
    # 计算频域中的能量
    freq_domain_energy = torch.abs(shifted_freq_domain_tensor) ** power # b, h * w * c, k, k
    freq_domain_energy = freq_domain_energy.reshape(b, h, w, c, win_size, win_size)
    # low_freq_energy = freq_domain_energy[..., win_size//2 - win_size//4:win_size//2 + win_size//4, win_size//2 - win_size//4:win_size//2 + win_size//4].sum((-3, -2, -1))
    # low_freq_energy = freq_domain_energy[..., win_size//2 - win_size//center_rate:win_size//2 + win_size//center_rate, win_size//2 - win_size//center_rate:win_size//2 + win_size//center_rate].sum((-3, -2, -1))
    if return_mode in ('high', 'low', 'high_ratio', 'low_ratio'):
        assert isinstance(lp_size, int) 
        if mask_mode == 'cycle':
            M = generate_circle_mask(win_size // lp_size, win_size,)
            M = torch.Tensor(M).reshape(1, 1, win_size, win_size)
        elif mask_mode == 'rect':
            M = torch.zeros((win_size, win_size))
            # M[win_size//2 - win_size//lp_size:win_size//2 + win_size//lp_size, win_size//2 - win_size//lp_size:win_size//2 + win_size//lp_size] = 1.0
            M[round(win_size/2 - win_size/lp_size):round(win_size/2 + win_size/lp_size), round(win_size/2 - win_size/lp_size):round(win_size/2 + win_size/lp_size)] = 1.0
        else: raise NotImplementedError
        low_freq_energy = (freq_domain_energy[..., :, :] * M).sum((-3, -2, -1))
        total_energy = freq_domain_energy.sum((-3, -2, -1)) # b, h * w * c
        hi_freq_energy = (total_energy - low_freq_energy)
        if return_mode == 'high':
            return hi_freq_energy.squeeze()
        elif return_mode == 'low':
            return low_freq_energy.squeeze()
        elif return_mode == 'high_ratio':
            return ((total_energy - low_freq_energy) / (total_energy + 1e-8)).squeeze()
        elif return_mode == 'low_ratio':
            return (low_freq_energy / (total_energy + 1e-8)).squeeze()
        else:
            raise NotImplementedError('high', 'low')
    elif return_mode in ('all'):
        assert isinstance(lp_size, list) 
        res = []
        pre = 0
        if len(lp_size) == 0:
            for i in list(range(1, win_size // 2))[::-1]:
                if mask_mode == 'cycle':
                    M = generate_circle_mask(i, win_size,)
                    M = torch.Tensor(M).reshape(1, 1, win_size, win_size)
                elif mask_mode == 'rect':
                    M = torch.zeros((win_size, win_size))
                    M[win_size//2 - i:win_size//2 + i, win_size//2 - i:win_size//2 + i] = 1.0
                else: raise NotImplementedError
                print(i, sum(M))
                low_freq_energy = (freq_domain_energy[..., :, :] * M).sum((-3, -2, -1))
                total_energy = freq_domain_energy.sum((-3, -2, -1)) # b, h * w * c
                hi_freq_energy = (total_energy - low_freq_energy)
                res.append(hi_freq_energy - pre)
                pre = hi_freq_energy
        else:
            for i in range(len(lp_size)):
                if mask_mode == 'cycle':
                    M = generate_circle_mask(win_size // lp_size[i], win_size,)
                    M = torch.Tensor(M).reshape(1, 1, win_size, win_size)
                elif mask_mode == 'rect':
                    M = torch.zeros((win_size, win_size))
                    M[win_size//2 - win_size//lp_size[i]:win_size//2 + win_size//lp_size[i], win_size//2 - win_size//lp_size[i]:win_size//2 + win_size//lp_size[i]] = 1.0
                else: raise NotImplementedError
                print(lp_size[i], sum(sum(M)))
                low_freq_energy = (freq_domain_energy[..., :, :] * M).sum((-3, -2, -1))
                total_energy = freq_domain_energy.sum((-3, -2, -1)) # b, h * w * c
                hi_freq_energy = (total_energy - low_freq_energy)
                res.append(hi_freq_energy - pre)
                pre = hi_freq_energy
        res.append(low_freq_energy)
        res = [i.squeeze() for i in res]
        return torch.stack(res[::-1], dim=0)

class AliasingMapGener(nn.Module):
    """
    lp_size = 4 -> 1/4, half of highest frequency
    """
    def __init__(self, win_size=5, power=2, channel_reduction=4, g_sigma=2, lp_size=4, mask_mode='rect', return_mode='high_ratio'):
        super().__init__()
        G = generate_gaussian_kernel(win_size, sigma=g_sigma)
        G = torch.Tensor(G).reshape(1, 1, win_size, win_size)
        self.register_buffer('G', G)
        self.channel_reduction = channel_reduction
        self.win_size = win_size
        self.return_mode = return_mode
        self.power = power
        if mask_mode == 'cycle':
            M = generate_circle_mask(win_size // lp_size, win_size,)
            M = torch.Tensor(M).reshape(1, 1, win_size, win_size)
        elif mask_mode == 'rect':
            M = torch.zeros((win_size, win_size))
            # M[win_size//2 - win_size//lp_size:win_size//2 + win_size//lp_size, win_size//2 - win_size//lp_size:win_size//2 + win_size//lp_size] = 1.0
            M[round(win_size/2 - win_size/lp_size):round(win_size/2 + win_size/lp_size), round(win_size/2 - win_size/lp_size):round(win_size/2 + win_size/lp_size)] = 1.0
        self.register_buffer('M', M)
    
    def forward(self, tensor):
        # return checkpoint(self._forward, tensor)
        return self._forward(tensor)

    def _forward(self, tensor):
        b, c, h, w = tensor.shape
        tensor = tensor.reshape(b, c // self.channel_reduction, self.channel_reduction, h, w)
        tensor = tensor.mean(-3)
        b, c, h, w = tensor.shape
        tensor = F.unfold(F.pad(tensor, pad=[self.win_size//2] * 4, mode='replicate'),
                        kernel_size=self.win_size, dilation=1, padding=0, stride=1)
        tensor = tensor.reshape(b, -1, h, w)
        tensor = tensor.permute(0, 2, 3, 1) # b, h, w, cxkxk
        tensor = tensor.reshape(b, h * w * c, self.win_size, self.win_size) # b, h * w * c, k, k
        # G = generate_gaussian_kernel(win_size, sigma=g_sigma)
        # G = torch.Tensor(G).reshape(1, 1, win_size, win_size)
        tensor = self.G * tensor
        # 将输入tensor转换为频域
        freq_domain_tensor = torch.fft.fftn(tensor, dim=(-2, -1))  # 对最后两维进行傅里叶变换
        # 将低频部分移动到频域中心
        shifted_freq_domain_tensor = torch.fft.fftshift(freq_domain_tensor, dim=(-2, -1))
        # 计算频域中的能量
        freq_domain_energy = torch.abs(shifted_freq_domain_tensor) ** self.power # b, h * w * c, k, k
        freq_domain_energy = freq_domain_energy.reshape(b, h, w, c, self.win_size, self.win_size)
        # low_freq_energy = freq_domain_energy[..., win_size//2 - win_size//4:win_size//2 + win_size//4, win_size//2 - win_size//4:win_size//2 + win_size//4].sum((-3, -2, -1))
        # low_freq_energy = freq_domain_energy[..., win_size//2 - win_size//center_rate:win_size//2 + win_size//center_rate, win_size//2 - win_size//center_rate:win_size//2 + win_size//center_rate].sum((-3, -2, -1))
        if self.return_mode in ('high', 'low', 'high_ratio', 'low_ratio'):
            low_freq_energy = (freq_domain_energy[..., :, :] * self.M).sum((-2, -1))
            low_freq_energy = low_freq_energy.permute(0, 3, 1, 2)
            total_energy = freq_domain_energy.sum((-2, -1)) # b, h * w * c
            total_energy = total_energy.permute(0, 3, 1, 2)
            # low_freq_energy = (freq_domain_energy[..., :, :] * self.M).sum((-3, -2, -1))
            # total_energy = freq_domain_energy.sum((-3, -2, -1)) # b, h * w * c
            hi_freq_energy = (total_energy - low_freq_energy)
            # print(low_freq_energy.shape)
            if self.return_mode == 'high':
                # return hi_freq_energy.squeeze()
                return hi_freq_energy
            elif self.return_mode == 'low':
                # return low_freq_energy
                return low_freq_energy.squeeze()
            elif self.return_mode == 'high_ratio':
                # return ((total_energy - low_freq_energy) / (total_energy + 1e-8)).squeeze()
                return ((total_energy - low_freq_energy) / (total_energy + 1e-8))
            elif self.return_mode == 'low_ratio':
                # return (low_freq_energy / (total_energy + 1e-8)).squeeze()
                return (low_freq_energy / (total_energy + 1e-8))
            else:
                raise NotImplementedError('high', 'low')
    
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


class SaliencySampler(nn.Module):
    def __init__(self, feat_dim=256, stride=2, padding_size=31, fwhm=32, dilation=1, use_checkpoint=True, detach=False, align_corners=True,
                sampler_mode='avgpsp_semantic_edge', 
                padding_mode='reflect', # better than replicate
                # padding_mode='replicate', 
                psp_ratio=4, log_dir=None):
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
                PSP(feat_dim, feat_dim // psp_ratio) if use_psp else nn.Identity(),
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
    
    def reverse_with_inv_grid(self, sampled_feat, grid, h, w):
        # grid_inv = self.create_grid_inv(grid, h, w)
        # invalid_mask = grid_inv[:, 0:1, :, :].isnan().repeat(1, sampled_feat.size(1), 1, 1) # b, 1, h, w
        # reversed_feat_with_grid_inv = F.grid_sample(sampled_feat, grid_inv.permute(0, 2, 3, 1).float(), mode='bilinear', align_corners=True)
        # vis_reversed_feats = reversed_feat_with_grid_inv.detach()
        # self.writer.add_images(f"reversed_feats_with_grid{h}x{w}", ((vis_reversed_feats.mean(dim=1, keepdim=True) - vis_reversed_feats.min()) / (vis_reversed_feats.max() - vis_reversed_feats.min())) * 64., global_step=None, walltime=None, dataformats='NCHW')
        # # print(invalid_mask.float().mean())
        # reversed_feat = self.reverse(sampled_feat, grid, h, w)
        # reversed_feat_with_grid_inv[invalid_mask] = reversed_feat[invalid_mask]
        # vis_reversed_feats = reversed_feat_with_grid_inv.detach()
        # self.writer.add_images(f"reversed_feats{h}x{w}", ((vis_reversed_feats.mean(dim=1, keepdim=True) - vis_reversed_feats.min()) / (vis_reversed_feats.max() - vis_reversed_feats.min())) * 64., global_step=None, walltime=None, dataformats='NCHW')
        # return reversed_feat_with_grid_inv
        inv_grid = invert_nonseparable_grid(grid.permute(0, 2, 3, 1), h, w, align_corners=self.align_corners)
        reversed_feat = F.grid_sample(sampled_feat, inv_grid, mode='bilinear', align_corners=self.align_corners)
        return reversed_feat

    def reverse(self, sampled_feat, grid, h, w):
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

    def reverse_grid(self, grid, h, w):
        inv_grid = invert_nonseparable_grid(grid.permute(0, 2, 3, 1), h, w, align_corners=self.align_corners)
        return inv_grid.permute(0, 3, 1, 2)

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
        if DCT_ANALYSIS:
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

from math import floor, ceil
from typing import List, Union

def invert_nonseparable_grid(grid, H, W, align_corners=True):
    grid = grid.clone()
    device = grid.device
    # _, _, H, W = input_shape
    B, grid_H, grid_W, _ = grid.shape # B, H, W, 2, \in (-1, 1), xy
    # assert B == input_shape[0]
    # print(grid.permute(0, 3, 1, 2))
    # print( H, W)
    eps = 1e-8
    if align_corners:
        grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
        grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    else:
        # grid \in [+1/2H, H-1/2H]
        grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * W 
        grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * H     
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    inverse_grid = 2 * max(H, W) * torch.ones(
        (B, H, W, 2), dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203
        p11 = grid[b, 1:  ,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).type(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        v11 = p11 - ref

        min_x = int(floor(min(v00[:, 0].min(), v10[:, 0].min()) - eps))
        max_x = int(ceil(max(v01[:, 0].max(), v11[:, 0].max()) + eps))
        min_y = int(floor(min(v00[:, 1].min(), v01[:, 1].min()) - eps))
        max_y = int(ceil(max(v10[:, 1].max(), v11[:, 1].max()) + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T

        # each of these is  ((grid_H - 1)*(grid_W - 1)) x 2
        vb = v10 - v00
        vc = v01 - v00
        vd = v00 - v10 - v01 + v11

        vx = pts.permute(1, 0).unsqueeze(0)  # 1 x (x_range*y_range) x 2
        Ma = v00.unsqueeze(1) - vx  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range) x 2

        vc_cross_vd = (vc[:, 0] * vd[:, 1] - vc[:, 1] * vd[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        vc_cross_vb = (vc[:, 0] * vb[:, 1] - vc[:, 1] * vb[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        Ma_cross_vd = (Ma[:, :, 0] * vd[:, 1].unsqueeze(1) - Ma[:, :, 1] * vd[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        Ma_cross_vb = (Ma[:, :, 0] * vb[:, 1].unsqueeze(1) - Ma[:, :, 1] * vb[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        qf_a = vc_cross_vd.expand(*Ma_cross_vd.shape)
        qf_b = vc_cross_vb + Ma_cross_vd
        qf_c = Ma_cross_vb

        mu_neg = -1 * torch.ones_like(Ma_cross_vd)
        mu_pos = -1 * torch.ones_like(Ma_cross_vd)
        mu_linear = -1 * torch.ones_like(Ma_cross_vd)

        nzie = (qf_a.abs() > 1e-10).expand(*Ma_cross_vd.shape)

        disc = (qf_b[nzie]**2 - 4 * qf_a[nzie] * qf_c[nzie]) ** 0.5
        mu_pos[nzie] = (-qf_b[nzie] + disc) / (2 * qf_a[nzie])
        mu_neg[nzie] = (-qf_b[nzie] - disc) / (2 * qf_a[nzie])
        mu_linear[~nzie] = qf_c[~nzie] / qf_b[~nzie]

        mu_pos_valid = torch.logical_and(mu_pos >= 0, mu_pos <= 1)
        mu_neg_valid = torch.logical_and(mu_neg >= 0, mu_neg <= 1)
        mu_linear_valid = torch.logical_and(mu_linear >= 0, mu_linear <= 1)

        mu = -1 * torch.ones_like(Ma_cross_vd)
        mu[mu_pos_valid] = mu_pos[mu_pos_valid]
        mu[mu_neg_valid] = mu_neg[mu_neg_valid]
        mu[mu_linear_valid] = mu_linear[mu_linear_valid]

        lmbda = -1 * (Ma[:, :, 1] + mu * vc[:, 1:2]) / (vb[:, 1:2] + vd[:, 1:2] * mu)  # noqa: E501

        unwarped_pts = torch.stack((lmbda, mu), dim=0)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501
    if align_corners:
        inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
        inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    else:
        inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W) * 2.0 - 1.0  # noqa: E501
        inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H) * 2.0 - 1.0  # noqa: E501
        # grid[:, :, :, 0] = 0.5 * (grid[:, :, :, 0] + 1) - 0.5 / w
        # grid[:, :, :, 1] = 0.5 * (grid[:, :, :, 1] + 1) - 0.5 / h
        # grid[:, :, :, 0] = grid[:, :, :, 0] * w
        # grid[:, :, :, 1] = grid[:, :, :, 1] * h
    # print(inverse_grid.permute(0, 3, 1, 2))
    clamp_value_x = 1 if align_corners else (1 - 0.5 / W * 2) 
    clamp_value_y = 1 if align_corners else (1 - 0.5 / H * 2) 
    # print(clamp_value_x, clamp_value_y)
    inverse_grid[..., 0] = torch.clamp(inverse_grid[..., 0],min=-clamp_value_x, max=clamp_value_x) #将里面的值的范围控制在[-1,1]
    inverse_grid[..., 1] = torch.clamp(inverse_grid[..., 1],min=-clamp_value_y, max=clamp_value_y)
    return inverse_grid.detach()

def torch_patchwise_dct(img, log=False):
    import torch_dct as dct
    # x = torch.randn(128, 128)
    x = torch.Tensor(img)
    assert x.dim() == 4
    b, c, h, w = x.shape
    # print(x.shape)
    # print(x.dim())
    # print(h - (h%8), w - (w%8))
    x = x[:, :, :h - (h%8), :w - (w%8)]
    x = F.unfold(x, kernel_size=(8, 8), stride=8, dilation=1)
    print(x.shape)
    x = x.reshape(b * c, 8, 8, -1).permute(0, 3, 1, 2)
    x = x.reshape(-1, 8, 8)
    # print(x.shape)

    X = dct.dct_2d(x)   # DCT-II done through the last dimension
    # print(X)
    # y = dct.idct_2d(X)  # scaled DCT-III done through the last dimension
    # print(torch.abs(x - y).sum())
    # print(torch.abs(x - y).mean())
    # assert (torch.abs(x - y)).mean() < 1e-10  # x == y within numerical tolerance
    # X_log = X.abs().log()
    # X = X_log
    X = X.abs()
    X = X.mean(dim=0)
    if log: 
        X = X.log()
        X[X < 0] = 0
    # X_log[X < 0] *= -1
    # X = (X - X.min()) / (X.max() - X.min()) * 255
    return X

class DCTSaver:
    def __init__(self, save_path='dct_analysis/zigzag_img2_8x_dct.npy'):
        self.dct_list = []
        self.save_path = save_path

    def save(self, x, print_avg=True, log=False):
        self.dct_list.append(x)
        avg = torch.stack(self.dct_list, dim=0).mean(dim=0)
        if log:
            avg = avg.log()
            avg[avg < 0] = 0
        avg = avg.numpy()
        if print_avg:print(self.save_path, len(self.dct_list), '\n', avg)
        np.save(self.save_path, avg)

DCT_ANALYSIS = False
# DCT_ANALYSIS = True
if DCT_ANALYSIS:
    uniform_2x_dct = DCTSaver(save_path='dct_analysis/uniform_2x_dct.npy')
    uniform_4x_dct = DCTSaver(save_path='dct_analysis/uniform_4x_dct.npy')
    uniform_8x_dct = DCTSaver(save_path='dct_analysis/uniform_8x_dct.npy')
    adaptive_2x_dct = DCTSaver(save_path='dct_analysis/adaptive_2x_dct.npy')
    adaptive_4x_dct = DCTSaver(save_path='dct_analysis/adaptive_4x_dct.npy')
    adaptive_8x_dct = DCTSaver(save_path='dct_analysis/adaptive_8x_dct.npy')
    original_dct = DCTSaver(save_path='dct_analysis/original_dct.npy')

def test_torch_patchwise_dct():
    s = DCTSaver()
    for i in range(10):
        x = torch.rand(4, 16, 64, 64)
        dct = torch_patchwise_dct(x)
        s.save(dct, True)

def dct_analysis():
    pass

def test_coordinate_generate():
    grid_size = 7
    padding_size = 0
    global_size = grid_size + padding_size * 2
    P_basis = torch.zeros(2, grid_size + 2 * padding_size, grid_size + 2 * padding_size)
    for k in range(2):
        for i in range(global_size):
            for j in range(global_size):
                #k = 0时,为(j-self.padding_size)/(self.grid_size-1.0)
                #k=1时,为(i-self.padding_size)/(self.grid_size-1.0)
                #k=0时,每一行都相同，前[0:30]是[-1,0], [30:60]是[0,1],[60:]是[1,2]
                ##k=1时,每一列都相同，前[0:30]是[-1,0], [30:60]是[0,1],[60:]是[1,2]
                P_basis[k,i,j] = k*(i - padding_size)/(grid_size - 1.0)+(1.0 - k)*(j- padding_size)/(grid_size-1.0)
    print(P_basis)
    x = torch.Tensor(list(range(-padding_size, grid_size + padding_size, 1))) / (grid_size - 1.0)
    grid = torch.stack(torch.meshgrid(x, x)[::-1], dim=0)
    print(grid)
    print(grid.shape)
    print(P_basis == grid)
    print(F.interpolate(grid.unsqueeze(0), size=(7, 7), mode='bilinear', align_corners=True))

    grid = grid.unsqueeze(0)
    print(F.grid_sample(grid, grid.permute(0, 2, 3, 1) * 2 - 1.0, mode='bilinear', align_corners=True, padding_mode="border"))

    x = torch.Tensor(list(range(-padding_size, grid_size + padding_size, 1))) / (grid_size) + 0.5 / grid_size
    not_align_grid = torch.stack(torch.meshgrid(x, x)[::-1], dim=0)
    print('not_align_grid', not_align_grid)
    not_align_grid = not_align_grid.unsqueeze(0)
    print(F.grid_sample(not_align_grid, not_align_grid.permute(0, 2, 3, 1) * 2 - 1.0, mode='bilinear', align_corners=False, padding_mode="border"))
    not_align_grid = not_align_grid - 0.5 / grid_size
    not_align_grid = not_align_grid * grid_size
    print('not_align_grid', not_align_grid)


def test_SaliencySampler():
    ss = SaliencySampler(feat_dim=3, padding_size=2, stride=1, fwhm=1)
    x = torch.rand(1, 1, 8, 8)
    # feature, feature_sampled, saliency = ss._test(x)
    # feature, feature_sampled, saliency = ss(x)
    img = cv2.imread("test_imgs/2012_004331.jpg")
    img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
    print(img.shape)
    saliency = torch.rand(1, 1, 64, 64)
    saliency[:, :, 10, 25] = 100
    # img, img_sampled, saliency, grid= ss(img)
    # saliency = torch.Tensor([[1, 1, 1, 1, 1], [1, 100000000, 1, 1, 1], [1, 1, 1, 1, 1]])[None, None]
    # saliency /= saliency.sum()
    _, _, h, w = img.shape
    img, img_sampled, saliency, grid = ss.sample_with_saliency(img, saliency, h, w)
    reversed_img = ss.reverse(img_sampled, grid, h, w)
    # print(ss.filter.weight.requires_grad)
    exit()
    
    print("saliency.std", saliency.std())
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    print("saliency", saliency)
    saliency *= 255
    print((reversed_img - img).abs().mean())
    cv2.imwrite('test_imgs/interp2d_test.jpg', img.squeeze().permute(1, 2, 0).numpy())
    cv2.imwrite('test_imgs/interp2d_saliency_test.jpg', saliency[0].permute(1, 2, 0).detach().numpy())
    cv2.imwrite('test_imgs/interp2d_sampled_test.jpg', img_sampled.squeeze().permute(1, 2, 0).detach().numpy())
    cv2.imwrite('test_imgs/interp2d_reversed_test.jpg', reversed_img.squeeze().permute(1, 2, 0).detach().numpy())

from mmcv.ops.carafe import carafe, Tensor, CARAFEPack
class NonUniformInterpolation(nn.Module):
    """A unified package of CARAFE upsampler that contains: 1) channel
    compressor 2) content encoder 3) CARAFE op.

    Official implementation of ICCV 2019 paper
    `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 padding_mode = 'replicate',
                 kernel_size=3):
        super().__init__()
        self.padding_mode = padding_mode

        self.kernel_size= kernel_size
        self.sigma = nn.parameter.Parameter(torch.tensor(.1), requires_grad=True)
        self.temp = nn.parameter.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, feat, non_uniform_grid, uniform_grid, up_scale, use_checkpoint=False) -> Tensor:
        if use_checkpoint:
            return checkpoint(self._forward, feat, non_uniform_grid, uniform_grid, up_scale)
        else:
            return self._forward(feat, non_uniform_grid, uniform_grid, up_scale)
        
    def _forward(self, feat, non_uniform_grid, uniform_grid, up_scale) -> Tensor:
        assert non_uniform_grid.size(-1) == feat.size(-1)
        assert non_uniform_grid.size(-2) == feat.size(-2)
        b, c, h, w = feat.shape
        feat_pad = F.pad(feat, pad=[(self.kernel_size // 2) * up_scale] * 4, mode=self.padding_mode)
        non_uniform_grid_pad = F.pad(non_uniform_grid, pad=[(self.kernel_size // 2) * up_scale] * 4, mode=self.padding_mode)
        # B x (C x ksize x ksize) x H // stride x W //stride
        feat_pad = F.unfold(feat_pad, kernel_size=(self.kernel_size, self.kernel_size), stride=1, dilation=up_scale)
        feat_pad = feat_pad.reshape(b, -1, self.kernel_size ** 2, h, w)
        non_uniform_grid_pad = F.unfold(non_uniform_grid_pad, kernel_size=(self.kernel_size, self.kernel_size), stride=1, dilation=up_scale)
        non_uniform_grid_pad = non_uniform_grid_pad.reshape(b, -1, self.kernel_size ** 2, h, w)
        # print(uniform_grid.shape)
        uniform_grid = uniform_grid.reshape(b, -1, 1, h, w)
        offset = non_uniform_grid_pad - uniform_grid
        # print(up_scale)
        # print(offset[0, :, :, h // 2, w // 2])
        ditance = torch.exp(-4 * np.log(2) * ((offset[:, 0:1])**2 + (offset[:, 1:2])**2) / self.sigma ** 2)
        # print(ditance[0, :, :, h // 2, w // 2])
        mask = (self.temp * ditance).softmax(dim=-3)
        # print(mask[0, :, :, h // 2, w // 2])
        out = mask * feat_pad
        out = out.sum(dim=2).reshape(b, -1, h, w)
        return out

# class ConvSaliencySampler(nn.Module):
class ConvSaliencySampler(SaliencySampler):
    def __init__(self, kernel_size, c_out=None, sampler_mode='conv3', coord_mode='yx', out_mode='abs_offset', **kwargs):
        super().__init__(**kwargs)
        del self.saliency_conv
        if c_out is None: c_out = kernel_size ** 2
        if 'conv' in sampler_mode:
            use_avg = ('avg' in sampler_mode)
            use_conv3 =  ('3' in sampler_mode)
            self.saliency_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size= 2 * self.stride + 1, stride= self.stride, padding= self.stride) if use_avg else nn.Identity(),
                nn.Conv2d(self.feat_dim, c_out, kernel_size=3, padding=self.dilation, dilation= self.dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(self.feat_dim, c_out, kernel_size=1, padding=0, stride=1)
            )
        elif 'lhpf' in sampler_mode: # learnable high pass filter
            use_psp = ('psp' in sampler_mode)
            use_conv3 = ('3' in sampler_mode)
            use_avg = ('avg' in sampler_mode)
            use_softpool = ('softpool' in sampler_mode)
            print(f"*** lhpf | use_psp:{use_psp} | use_avg:{use_avg} | use_softpool:{use_softpool} ")
            if use_avg: 
                assert not use_softpool
                POOL = nn.AvgPool2d(kernel_size= 2 *  self.stride + 1, stride=self.stride, padding=self.stride)
            elif use_softpool:
                POOL = SoftPool2d(kernel_size = 2 *  self.stride + 1, stride=self.stride, padding=self.stride)
            else:
                POOL = nn.Identity()
            self.saliency_conv = nn.Sequential(
                POOL,
                LHPFConv3(channels=self.feat_dim, stride=1, padding=1, residual=True), 
                # LHPFConv(channels=self.feat_dim, kernel_size=5, stride=1, padding=2), 
                # nn.SyncBatchNorm(self.feat_dim), 
                # nn.ReLU(inplace=True),
                # nn.Conv2d(self.feat_dim, 1, kernel_size=3, padding=1, stride=1),
                PSP(self.feat_dim, self.feat_dim // 4) if use_psp else nn.Identity(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(self.feat_dim * (2 if use_psp else 1), c_out, kernel_size=3, padding=self.dilation, dilation=self.dilation, stride=1) if use_conv3 \
                    else  nn.Conv2d(self.feat_dim * (2 if use_psp else 1), c_out, kernel_size=1, padding=0, stride=1)
            )
        # self.saliency_conv = nn.Conv2d(self.feat_dim, c_out, kernel_size=kernel_size, padding=(kernel_size // 2) * self.dilation, stride=self.stride, dilation=self.dilation, bias=True)
        # self.channel_conv = nn.Conv2d(self.feat_dim, kernel_size ** 2, kernel_size=1, padding=0, stride=1, bias=True)
        # self.saliency_conv.weight.data.zero_()
        # self.saliency_conv.bias.data.zero_()
        # self.channel_conv.weight.data.zero_()
        # self.channel_conv.bias.data.zero_()
        # self.eps = 1e-8
        self.eps = 1e-16
        self.coord_mode = coord_mode
        self.out_mode = out_mode # abs_offset for deformconv, rel_grid for grid_sample
        del self.filter
        self.init_gaussian_filter()

    def init_gaussian_filter(self):
        gaussian_weights = torch.Tensor(makeGaussian(2 * self.padding_size + 1, fwhm = self.fwhm))
        self.gaussian_weights = nn.Parameter(gaussian_weights[None, None, :], requires_grad=False)
        # print("gaussian_weights", gaussian_weights)
        # self.filter = nn.Conv2d(1, 1, kernel_size = (2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        # self.filter.weight[0].data[:,:,:] = gaussian_weights
        # self.filter.weight.requires_grad = False
        # print(self.gaussian_weights)

    def Filter(self, x):
        # print(self.gaussian_weights)
        return F.conv2d(x, weight=self.gaussian_weights, bias=None, stride=1, padding=0, dilation=1, groups=1)

    def forward(self, feature, sample_with_grid=True):
        # _, c, h, w = feature.shape
        # nh, nw = h // self.stride, w // self.stride
        # saliency = self.saliency_conv(F.relu(feature, inplace=False)) #得到saliency map,channels=1,即[batch_size, c, h, w]
        # saliency = self.saliency_conv(feature) #得到saliency map,channels=1,即[batch_size, c, h, w]
        # if self.saliency_conv is None:
        #     b, _, h, w = feature.shape
        #     feature_sampled = feature
        #     saliency = None
        #     grid = self.gen_coord_grid_without_pad(h, w, device=feature.device)
        #     grid = grid[None, ].repeat(b, 1, 1, 1) * 2 - 1
        #     return feature, feature_sampled, saliency, grid
            
        # if self.use_checkpoint:
        #     saliency = checkpoint(self.saliency_conv, feature) #得到saliency map,channels=1,即[batch_size, c, h, w]
        # else:
        #     saliency = self.saliency_conv(feature)
        # t = self.channel_conv(F.adaptive_avg_pool2d(feature, 1))
        # saliency = self.saliency_conv(feature) * t
        # print('padding', self.saliency_conv.padding)
        saliency = self.saliency_conv(feature)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        # saliency = saliency.view(b, c, h * w).sigmoid().view(b, c, h, w) * h * w
        offsets = self.create_conv_offset(saliency, h, w) # b, h, w, 2
        # print(feature.shape, offsets.shape)
        # print(self.saliency_conv.dilation, self.saliency_conv.kernel_size)
        # print(offsets.min(), offsets.max())
        return offsets
        # return feature, feature_sampled, reversed_feat, saliency, grid
    
    def gen_coord_grid(self, h, w, device=None):
        """
        return: [2, h, w]
        """
        if self.align_corners:
            # x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / (w - 1.0)
            # y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / (h - 1.0)
            x = torch.arange(-self.padding_size, w + self.padding_size, 1) / (w - 1.0)
            y = torch.arange(-self.padding_size, h + self.padding_size, 1) / (h - 1.0)
        else:
            x = torch.arange(-self.padding_size, w + self.padding_size, 1) / w + 0.5 / w
            y = torch.arange(-self.padding_size, h + self.padding_size, 1) / h + 0.5 / h
        coord_grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0)
        # print(coord_grid.shape)
        # coord_grid = torch.stack(torch.meshgrid([x, y]), dim=0).transpose(1, 2)
        # print(coord_grid.shape)
        if device is not None: coord_grid = coord_grid.to(device)
        return coord_grid
    
    def create_conv_offset(self, s, h, w):
        """
        x: saliency in [b, 1, h, w]
        """
        #x : 相当于论文中等式2和等式3的S(x',y')* k((x,y),(x',y')),是权重
        #P : 相当于论文中等式2和等式3的[x',y']
        # P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)
        # P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cpu(),requires_grad=False)
        # x = nn.ReplicationPad2d(self.padding_size)(x) #避免等式2和3偏向于图像中心的采样偏好
        # x = nn.ReflectionPad2d(self.padding_size)(x)
        # print(self.filter.weight.requires_grad)
        # print(self.gaussian_weights)
        s = F.pad(s, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size
        # P = torch.zeros(1, 2, padded_h, padded_w).to(x.device)
        # P[0,:,:,:] = self.gen_coord_grid(h, w, device=x.device)
        P = self.gen_coord_grid(h, w, device=s.device).reshape(1, 2, padded_h, padded_w)
        # print(self.coord_grid.requires_grad)
        #从[1, 2, 91, 91]扩展为[batch_size, 2, 91, 91]
        P = P.expand(s.size(0), -1, padded_h, padded_w)
        P = P.repeat(1, s.size(1), 1, 1)
        # print("P", P[0, :, 0, 0])
        # print("P", P.shape)
        # print("P size is : ", P.size()) #torch.Size([5, 2, 91, 91])

        # x_cat = torch.stack((x, x), 1) #[batch_size, 2, 91, 91]
        # x_cat = x_cat.reshape(x.size(0), -1, padded_h, padded_w)
        # print("x_cat", x_cat[0, :, 0, 0])
        # print("x_cat size is : ", x_cat.size()) #torch.Size([5, 2, 91, 91])
        #得到的是论文中等式2的分母
        # print("x size is : ", x.size()) #torch.Size([5, 2, 91, 91])
        p_filter = self.Filter(s.view(-1, 1, padded_h, padded_w)) #输入[batch_size, 1, 91, 91]，输出[batch_size, 1, 31, 31]
        p_filter = p_filter.view(s.size(0), -1, h, w) #输入[batch_size, 1, 91, 91]，输出[batch_size, 1, 31, 31]
        # print("x", x.type())
        # print("p_filter", p_filter.type())
        # print("self.filter.weight", self.filter.weight)
        # print(self.filter.weight.requires_grad)
        # print("p_filter is : ", p_filter)
        # print("p_filter size is : ", p_filter.size()) #torch.Size([5, 1, 31, 31])

        '''
        #得到的是论文中等式2和等式3的分子
        x_mul = torch.mul(P, x_cat).view(-1, 1, padded_h, padded_w) #[batch_size*2, 1, 91, 91]
        # print("x_mul size is : ", x_mul.size()) #torch.Size([10, 1, 91, 91])
        #filter()输入[batch_size*2, 1, 91, 91], 输出[batch_size*2, 1, 31, 31]
        #然后重置为[batch_size, 2, 31, 31]
        all_filter = self.filter(x_mul).view(x.size(0), -1, h, w)
        # print("all_filter size is : ", all_filter.size()) #torch.Size([5, 2, 31, 31])

        # x_filter是u(x,y)的分子,y_filter是v(x,y)的分子
        x_filter = all_filter[:,0::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        y_filter = all_filter[:,1::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        # print("y_filter size is : ", y_filter.size()) #torch.Size([5, 1, 31, 31])
        '''
        x_filter = torch.mul(P[:,0::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        x_filter = self.Filter(x_filter).view(s.size(0), -1, h, w)
        y_filter = torch.mul(P[:,1::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        y_filter = self.Filter(y_filter).view(s.size(0), -1, h, w)

        #值的范围是[0,1]
        # x_filter = x_filter / p_filter #u(x,y)
        # y_filter = y_filter / p_filter #v(x,y)
        # print(x.min(), x.max(), x.mean())
        # x_filter = x_filter / (p_filter + 1e-16) #u(x,y)
        # y_filter = y_filter / (p_filter + 1e-16) #v(x,y)
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter + self.eps) #v(x,y)
        # print("x_filter is : ", x_filter.min(), x_filter.max())

        #将值的范围从[0,1]改为[-1,1]
        # xgrids = x_filter * 2 - 1
        # ygrids = y_filter * 2 - 1
        # print(x_filter.min(), x_filter.max())
        # (w - 1.0)
        if 'abs_offset' ==  self.out_mode:
            x_offsets = (x_filter - P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * w
            y_offsets = (y_filter - P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * h
        elif 'rel_grid' ==  self.out_mode:
            x_offsets = x_filter * 2.0 - 1.0
            y_offsets = y_filter * 2.0 - 1.0
        else:
            raise NotImplementedError
        # print( 'x_offsets', x_offsets.min(), x_offsets.max())
        # x_offsets = torch.clamp(x_offsets, min= -(w - 1), max= +(w - 1))
        # y_offsets = torch.clamp(y_offsets, min= -(h - 1), max= +(h - 1))
        # grid = torch.cat((x_offsets, y_offsets), 1) #[batch_size, 2, 31, 31]
        # grid = torch.cat((y_offsets, x_offsets), 1) #[batch_size, 2, 31, 31]
        if self.coord_mode == 'xy': # dcnv3, grid_sample
            grid = torch.stack((x_offsets, y_offsets), 2) #[batch_size, position, 2, h, w]
        elif self.coord_mode == 'yx': # dcnv2
            grid = torch.stack((y_offsets, x_offsets), 2) #[batch_size, position, 2, h, w]
        else:
            raise NotImplementedError
        grid = grid.reshape(s.size(0), -1, h, w) #[batch_size, position, 2, h, w]
        
        '''
        # print("ygrids max is : ", ygrids.max()) #tensor(1.0200, grad_fn=<MaxBackward1>)
        # print("ygrids min is : ", ygrids.min()) #tensor(-1.0502, grad_fn=<MinBackward1>)
        clamp_value_x = 1 if self.align_corners else (1 - 0.5 / w * 2) 
        clamp_value_y = 1 if self.align_corners else (1 - 0.5 / h * 2) 
        # print(clamp_value_x, clamp_value_y)
        xgrids = torch.clamp(xgrids,min=-clamp_value_x, max=clamp_value_x) #将里面的值的范围控制在[-1,1]
        ygrids = torch.clamp(ygrids,min=-clamp_value_y, max=clamp_value_y)

        xgrids = xgrids.view(x.size(0), -1, h, w)
        ygrids = ygrids.view(x.size(0), -1, h, w)
        '''
        # grid = torch.cat((xgrids, ygrids), 1) #[batch_size, 2, 31, 31]
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
        '''
        grid[:, x.size(1):, 0, :] = P[:, 1::2, self.padding_size, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, x.size(1):, -1, :] = P[:, 1::2, -self.padding_size-1, self.padding_size:-self.padding_size] * 2 - 1.0
        grid[:, :x.size(1), :, 0] = P[:, 0::2, self.padding_size:-self.padding_size, self.padding_size] * 2 - 1.0
        grid[:, :x.size(1), :, -1] = P[:, 0::2, self.padding_size:-self.padding_size, -self.padding_size-1] * 2 - 1.0
        '''
        return grid # in x, y
    
    def sample_with_grid(self, feature, grid):
        assert self.coord_grid == 'xy'
        assert self.out_mode == 'rel_grid'
        feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled
        
class SFAlignModule(nn.Module):
    def __init__(self, inplane, compress_ratio=4, outplane=None, flow_make_k=3, **kwargs):
        super().__init__()
        if outplane is None: outplane = inplane // compress_ratio
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=flow_make_k, padding=flow_make_k//2, bias=False)

    # def forward(self, x):
    def forward(self, hr_x, lr_x, use_checkpoint=True):
        low_feature, h_feature = hr_x, lr_x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return hr_x, h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, padding_mode='border')
        return output
    
class ASAlignModule(nn.Module):
    def __init__(self, inplane, compress_ratio=4, outplane=None, align_groups=1, radius=2, hr_flow=False):
        super().__init__()
        if outplane is None: outplane = inplane // compress_ratio
        self.down_lr = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_hr = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.align_groups = align_groups
        self.radius = radius
        # self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make_lr = ConvSaliencySampler(kernel_size=3, c_out=align_groups, coord_mode='xy', out_mode='rel_grid', 
                                             feat_dim=outplane * 2, 
                                             stride=1, 
                                             padding_size=self.radius + 1, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
                                             sampler_mode='conv3', 
                                             padding_mode='reflect', # better than replicate
                                            #  padding_mode='replicate'
                                             )
        self.hr_flow = hr_flow
        if self.hr_flow:
            self.flow_make_hr = ConvSaliencySampler(kernel_size=3, c_out=align_groups, coord_mode='xy', out_mode='rel_grid', 
                                                feat_dim=outplane * 2, 
                                                stride=1, 
                                                padding_size=self.radius + 1, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
                                                sampler_mode='conv3', 
                                                padding_mode='reflect', # better than replicate
                                                # padding_mode='replicate'
                                                )

    def forward(self, hr_x, lr_x, use_checkpoint=True):
        if use_checkpoint:
            return checkpoint(self._forward, hr_x, lr_x)
        else:
            return self._forward(hr_x, lr_x)
        
    def _forward(self, hr_x, lr_x):
        """
        low_feature, h_feature, low-level, high-level feature
        """
        # low_feature, h_feature= x
        lr_x_orign = lr_x
        h, w = hr_x.size()[2:]
        size = (h, w)
        hr_feat = self.down_hr(hr_x)
        lr_feat= self.down_lr(lr_x)
        lr_feat = F.interpolate(lr_feat, size=size, mode="bilinear", align_corners=False)
        cat_feat = torch.cat([lr_feat, hr_feat], 1)
        flow = self.flow_make_lr(cat_feat)
        # print(flow.shape)
        # print('flow', flow.shape, 'lr_x_orign', lr_x_orign.shape)
        lr_x = self.flow_warp(lr_x_orign, flow, size=size)
        if self.hr_flow: 
            hr_flow = self.flow_make_hr(cat_feat)
            hr_x = self.flow_warp(hr_x, hr_flow, size=size)
        return hr_x, lr_x

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        # n, c, h, w = input.size()
        # norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        # grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        # grid = grid + flow.permute(0, 2, 3, 1) / norm
        # grid = self.flow_make_lr(flow)
        grid = flow
        n, c, h, w = grid.size()
        # grid = grid.reshape(n, self.align_groups, -1, h, w).reshape(n * self.align_groups, -1, h, w)
        grid = grid.reshape(n * self.align_groups, -1, h, w)
        grid = grid.permute(0, 2, 3, 1)
        input = input.reshape(n * self.align_groups, -1, input.size(-2), input.size(-1))
        # print('input', input.shape)
        output = F.grid_sample(input, grid, padding_mode='border')
        # print('output', output.shape)
        output = output.reshape(n, -1, h, w)
        # print('output', output.shape)
        return output
    
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, CONV_LAYERS, modulated_deform_conv2d, ModulatedDeformConv2dPack
# from dcn_v2 import dcn_v2_conv

@CONV_LAYERS.register_module('DCN_ASv1')
class DCNv2_ASv1(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, only_center=True, radius=3, **kwargs):
        super().__init__(*args, **kwargs)
        # self.CA = ChannelAttention(in_planes=self.in_channels)
        # self.SA = SpatialAttention(kernel_size=7)
        self.only_center = only_center
        self.radius = radius
        # self.group = group
        if self.only_center:
            self.AS_offset = ConvSaliencySampler(
                kernel_size=self.kernel_size[0],
                c_out=self.deform_groups,
                # group=group,
                feat_dim=self.in_channels , 
                stride=self.stride, 
                # stride=1, 
                padding_size=self.radius, fwhm=self.radius, dilation=1, use_checkpoint=True, detach=False, align_corners=False,
                sampler_mode='conv3', 
                # padding_mode='reflect', # better than replicate
                padding_mode='replicate', 
                psp_ratio=4, log_dir=None)
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 2 * (self.kernel_size[0] * self.kernel_size[1] - 1),
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
        else:
            self.AS_offset = ConvSaliencySampler(
                kernel_size=self.kernel_size[0],
                c_out=self.deform_groups * self.kernel_size[0] * self.kernel_size[1],
                feat_dim=self.in_channels, 
                stride=self.stride, 
                # stride=1, 
                padding_size=self.radius, fwhm=self.radius, dilation=1, use_checkpoint=True, detach=False, align_corners=False,
                sampler_mode='conv3', 
                # padding_mode='reflect', # better than replicate
                padding_mode='replicate', 
                psp_ratio=4, log_dir=None)
        
        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 1 * (self.kernel_size[0] * self.kernel_size[1]),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

        if hasattr(self, 'conv_mask'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    def forward(self, x, test_offset=None):
        # x = self.CA(x) * x
        # x = self.SA(x) * x
        """
        # DCNv2-pytorch_1.9/src/cpu/dcn_v2_im2col_cpu.cpp（参考实现）
        # const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        # // offset的具体位置
        # const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        # const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        # const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        # const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        """
        if self.only_center:
            offset = self.conv_offset(x)
            b, c, h, w = offset.shape
            offset = offset.reshape(b, self.deform_groups, -1, h, w)
            # o1, o2 = torch.chunk(offset, 2, dim=2)

            AS_offset = self.AS_offset(x)
            # AS_offset = AS_offset.reshape(b, 1, -1, h, w).repeat(1, self.deform_groups, 1, 1, 1).contiguous()
            AS_offset = AS_offset.reshape(b, self.deform_groups, -1, h, w)
            # AS_o1, AS_o2 = torch.chunk(AS_offset, 2, dim=2)

            o_n = self.kernel_size[0] * self.kernel_size[1] - 1
            # o1_list = list(torch.chunk(o1, o_n, dim=2))
            # o2_list = list(torch.chunk(o2, o_n, dim=2))
            # print(o2_list[0].shape, AS_o1.shape)
            offset_list = list(torch.chunk(offset, o_n, dim=2))

            # o1 = torch.cat(o1_list[:o_n // 2] + [AS_o1] + o1_list[o_n // 2:], 1)
            # o1 = torch.cat(o2_list[:o_n // 2] + [AS_o2] + o2_list[o_n // 2:], 1)
            # o1 = torch.cat([*o1_list[:o_n // 2], AS_o1, *o1_list[o_n // 2:]], 2)
            # o2 = torch.cat([*o2_list[:o_n // 2], AS_o2, *o2_list[o_n // 2:]], 2)
            # offset = torch.cat((o1, o2), dim=2).reshape(b, -1, h, w)
            offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[o_n // 2:]], 2).reshape(b, -1, h, w)
        else:
            offset = self.AS_offset(x)

        # print(offset.shape)
        # print(x.shape)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        # print(mask)
        # if test_offset is not None: 
        #     offset = test_offset.float()
        #     mask.fill_(1.)
        # return dcn_v2_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deform_groups)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
    
@CONV_LAYERS.register_module('DCN_AS')
class DCNv2_AS(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, 
                 *args, 
                 only_center=True, radius=3, plus_center=False, offset_size=None, offset_kernel=None, offset_dilation=None, fp16=False, 
                 kernel_filter_wise=False, cr=8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.CA = ChannelAttention(in_planes=self.in_channels)
        # self.SA = SpatialAttention(kernel_size=7)
        print("====== DCNv2_AS settings ======")
        print("only_center:", only_center)
        print("radius:", radius)
        print("plus_center:", plus_center)
        print("offset_size:", offset_size)
        print("offset_kernel:", offset_kernel)
        print("offset_dilation:", offset_dilation)
        print("fp16:", fp16)
        print("====== DCNv2_AS settings ======")
        self.only_center = only_center
        self.radius = radius
        self.plus_center = plus_center
        self.fp16 = fp16
        assert self.kernel_size[0] == self.kernel_size[1]
        # self.group = group
        if offset_dilation is None:
            self.offset_dilation = self.dilation[0]
        else:
            self.offset_dilation = offset_dilation

        if offset_kernel is None: 
            self.offset_kernel = self.kernel_size[0]
            self.offset_padding = self.padding[0]
        else:
            self.offset_kernel = offset_kernel
            self.offset_padding = (self.offset_kernel // 2) * self.offset_dilation
        # print(self.offset_padding)
        if offset_size is None: 
            self.offset_size = self.kernel_size[0]
            self.use_offset_interp = False
        else:
            self.offset_size = offset_size
            self.use_offset_interp = True

            
        if self.only_center:
            self.AS_offset = ConvSaliencySampler(
                # kernel_size=self.kernel_size[0],
                kernel_size=self.offset_kernel,
                c_out=self.deform_groups,
                dilation=self.offset_dilation,
                # group=group,
                feat_dim=self.in_channels, 
                stride=self.stride, 
                # stride=1, 
                padding_size=self.radius + 1, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
                sampler_mode='avg_lhpf_conv', 
                coord_mode='yx',
                # padding_mode='reflect', # better than replicate
                padding_mode='replicate', 
                psp_ratio=4, log_dir=None)
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 2 * (self.offset_size * self.offset_size - 1),
                # self.deform_groups * 2 * (self.kernel_size[0] * self.kernel_size[1] - 1),
                kernel_size=self.offset_kernel,
                # kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.offset_padding,
                dilation=self.offset_dilation,
                bias=True)
        else:
            self.AS_offset = ConvSaliencySampler(
                kernel_size=self.offset_kernel,
                c_out=self.deform_groups * self.offset_size * self.offset_size,
                # c_out=self.deform_groups * self.kernel_size[0] * self.kernel_size[1],
                dilation=self.offset_dilation,
                feat_dim=self.in_channels, 
                stride=self.stride, 
                # stride=1, 
                padding_size=self.radius, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
                sampler_mode='conv3', 
                coord_mode='yx',
                # padding_mode='reflect', # better than replicate
                padding_mode='replicate', 
                psp_ratio=4, log_dir=None)
        
        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 1 * self.offset_size * self.offset_size,
            # self.deform_groups * 1 * (self.kernel_size[0] * self.kernel_size[1]),
            kernel_size=self.offset_kernel,
            stride=self.stride,
            padding=self.offset_padding,
            # padding=self.padding,
            dilation=self.offset_dilation,
            bias=True)
        # self.conv_offset_dilation = nn.Conv2d(
        #     self.in_channels,
        #     # self.deform_groups * 1 * self.offset_size * self.offset_size,
        #     self.deform_groups * 1,
        #     # self.deform_groups * 1 * (self.kernel_size[0] * self.kernel_size[1]),
        #     kernel_size=self.offset_kernel,
        #     stride=self.stride,
        #     padding=self.offset_padding,
        #     # padding=self.padding,
        #     dilation=self.offset_dilation,
        #     bias=True)
        
        self.kernel_filter_wise = kernel_filter_wise
        if kernel_filter_wise:
            self.global_fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(self.in_channels, self.in_channels // cr, kernel_size=1, stride=self.stride, padding=0, dilation=self.dilation, bias=True),
                nn.SyncBatchNorm(self.in_channels // cr),
                # nn.BatchNorm2d(attention_channel),
                nn.ReLU(True)
            )
            self.kernel_wise = nn.Conv2d(self.in_channels // cr, self.in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
            self.filter_wise = nn.Conv2d(self.in_channels // cr, self.in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
            # self.lowpart = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
            # self.sigmoid = HSigmoid()
            self.sigmoid = nn.Sigmoid()
            
            self.kernel_wise.weight.data.zero_()
            self.kernel_wise.bias.data.zero_()
            self.filter_wise.weight.data.zero_()
            self.filter_wise.bias.data.zero_()

        self.init_weights()
        print(self)

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

        if hasattr(self, 'conv_mask'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

        if hasattr(self, 'conv_offset_dilation'):
            self.conv_offset_dilation.weight.data.zero_()
            self.conv_offset_dilation.bias.data.zero_()

    def forward(self, x, test_offset=None):
        # x = self.CA(x) * x
        # x = self.SA(x) * x
        """
        # DCNv2-pytorch_1.9/src/cpu/dcn_v2_im2col_cpu.cpp（参考实现）
        # const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        # // offset的具体位置
        # const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        # const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        # const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        # const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        """
        # print(self.conv_offset.dilation)
        # print(self.conv_mask.dilation)
        # print(self.AS_offset.dilation)
        # offset_dilation = self.conv_offset_dilation(x).sigmoid()
        if self.kernel_filter_wise:
            global_feat = self.global_fc(x)
            kernel_att = self.sigmoid(self.kernel_wise(global_feat))
            filter_att = self.sigmoid(self.filter_wise(global_feat))
            x = x * kernel_att

        if self.only_center:
            offset = self.conv_offset(x)
            b, c, h, w = offset.shape
            offset = offset.reshape(b, self.deform_groups, -1, h, w)
            # offset_dilation = offset_dilation.reshape(b, self.deform_groups, -1, h, w)
            # offset = offset * offset_dilation
            # print('offset', offset.shape)
            # o1, o2 = torch.chunk(offset, 2, dim=2)

            AS_offset = self.AS_offset(x)
            # AS_offset = AS_offset.reshape(b, 1, -1, h, w).repeat(1, self.deform_groups, 1, 1, 1).contiguous()
            AS_offset = AS_offset.reshape(b, self.deform_groups, -1, h, w)
            # print('AS_offset', AS_offset.shape)
            # AS_o1, AS_o2 = torch.chunk(AS_offset, 2, dim=2)
            if self.plus_center: 
                # print('plus_center')
                offset += AS_offset.repeat(1, 1, c // self.deform_groups // 2 , 1, 1)

            o_n = self.offset_size * self.offset_size - 1
            # o1_list = list(torch.chunk(o1, o_n, dim=2))
            # o2_list = list(torch.chunk(o2, o_n, dim=2))
            # print(o2_list[0].shape, AS_o1.shape)
            offset_list = list(torch.chunk(offset, o_n, dim=2))

            # o1 = torch.cat(o1_list[:o_n // 2] + [AS_o1] + o1_list[o_n // 2:], 1)
            # o1 = torch.cat(o2_list[:o_n // 2] + [AS_o2] + o2_list[o_n // 2:], 1)
            # o1 = torch.cat([*o1_list[:o_n // 2], AS_o1, *o1_list[o_n // 2:]], 2)
            # o2 = torch.cat([*o2_list[:o_n // 2], AS_o2, *o2_list[o_n // 2:]], 2)
            # offset = torch.cat((o1, o2), dim=2).reshape(b, -1, h, w)
            # offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[o_n // 2:]], 2).reshape(b, -1, h, w)
            if self.use_offset_interp:
                offset = torch.stack([*offset_list[:o_n // 2], AS_offset, *offset_list[-(o_n // 2):]], 2) # B, DeformGroup, offset_size**2, 2, H, W
                offset = offset.permute(0, 1, 4, 5, 3, 2) # B, DeformGroup, H, W, 2, offset_size**2
                # print('offset', offset.shape)
                offset = offset.reshape(-1, 2, self.offset_size, self.offset_size)
                offset = F.interpolate(offset, size=(self.kernel_size[0], self.kernel_size[1]), mode='bilinear', align_corners = (1 == (self.offset_size // 2)))
                offset = offset.reshape(b, self.deform_groups, h, w, 2, self.kernel_size[0], self.kernel_size[1]) # b, self.deform_groups, h, w, 2, self.kernel_size[0], self.kernel_size[1]
                offset = offset.permute(0, 1, 5, 6, 4, 2, 3) # b, self.deform_groups,  self.kernel_size[0], self.kernel_size[1], 2, h, w
                offset = offset.reshape(b, -1, h, w)
            else:
                offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[-(o_n // 2):]], 2) # B, DeformGroup, offset_size * 2, H, W
                offset = offset.reshape(b, -1, h, w)
        else:
            offset = self.AS_offset(x)

        # print(offset.shape)
        # print(x.shape)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        # offset.fill_(0.0)
        # mask.fill_(1.0)
        if self.use_offset_interp:
            mask = mask.reshape(b, self.deform_groups, self.offset_size, self.offset_size, h, w)
            mask = mask.permute(0, 1, 4, 5, 2, 3) # b, deformgroup, h, w, o, o
            mask = mask.reshape(b, -1, self.offset_size, self.offset_size)
            mask = F.interpolate(mask, size=(self.kernel_size[0], self.kernel_size[1]), mode='bilinear', align_corners = (1 == (self.offset_size // 2)))
            mask = mask.reshape(b, self.deform_groups, h, w, self.kernel_size[0], self.kernel_size[1])
            mask = mask.permute(0, 1, 4, 5, 2, 3)# b, o, o, h, w,
            mask = mask.reshape(b, -1, h, w)
        # print(mask)
        # if test_offset is not None: 
        #     offset = test_offset.float()
        #     mask.fill_(1.)
        # return dcn_v2_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deform_groups)
        if self.fp16:
            x = modulated_deform_conv2d(x.float(), offset.float(), mask.float(), self.weight.float(), self.bias.float(),
                                        self.stride, self.padding,
                                        self.dilation, self.groups,
                                        self.deform_groups)
        else:
            x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                           self.stride, self.padding,
                                           self.dilation, self.groups,
                                           self.deform_groups)
        if self.kernel_filter_wise:
            x = x * filter_att
        return x

class FLC_Pooling(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, downsample=False, freq_thres=0.25):
        super(FLC_Pooling, self).__init__()
        assert freq_thres < (0.5 + 1e-8)
        assert freq_thres > 0.0
        self.downsample = downsample
        self.freq_thres = freq_thres

    def forward(self, x):
        if self.downsample:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
            return torch.fft.ifft2(torch.fft.ifftshift(low_part)).real
        else:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))
            mask = torch.zeros_like(low_part, device=low_part.device)
            # mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
            _, _, h, w = x.shape
            mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
            return torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real


@CONV_LAYERS.register_module('DCN_PreAS')
class DCNv2_PreAS(ModulatedDeformConv2dPack):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, 
                 *args, 
                 radius=3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        # self.AS = ConvSaliencySampler(
        #     kernel_size=self.kernel_size[0],
        #     c_out=self.deform_groups,
        #     dilation=self.dilation,
        #     # group=group,
        #     feat_dim=self.in_channels, 
        #     stride=self.stride, 
        #     # stride=1, 
        #     padding_size=self.radius + 1, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
        #     sampler_mode='conv3', 
        #     out_mode='rel_grid',
        #     coord_mode='xy',
        #     # padding_mode='reflect', # better than replicate
        #     padding_mode='replicate', 
        #     psp_ratio=4, log_dir=None)
        self.AS = SaliencySampler(feat_dim=self.in_channels, 
                                  stride=1, padding_size=self.radius + 1, fwhm=self.radius, 
                                  sampler_mode='conv3', align_corners=False)
        
    def forward(self, x):
        _, x, _, _ = self.AS(x)
        # AS_grid = self.AS(x)
        # x = self.AS.sample_with_grid(x, AS_grid)
        
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
    
@CONV_LAYERS.register_module('DCN_AS_PreAS')
class DCNv2_AS_PreAS(DCNv2_AS):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, 
                 *args, 
                 radius=3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        # self.AS = ConvSaliencySampler(
        #     kernel_size=self.kernel_size[0],
        #     c_out=self.deform_groups,
        #     dilation=self.dilation,
        #     # group=group,
        #     feat_dim=self.in_channels, 
        #     stride=self.stride, 
        #     # stride=1, 
        #     padding_size=self.radius + 1, fwhm=self.radius, use_checkpoint=False, detach=False, align_corners=False,
        #     sampler_mode='conv3', 
        #     out_mode='rel_grid',
        #     coord_mode='xy',
        #     # padding_mode='reflect', # better than replicate
        #     padding_mode='replicate', 
        #     psp_ratio=4, log_dir=None)
        del self.AS_offset
        self.AS = SaliencySampler(feat_dim=self.in_channels, 
                                  stride=1, padding_size=self.radius + 1, fwhm=self.radius, 
                                  sampler_mode='conv3', align_corners=False)
        
    def forward(self, x):
        ori_x = x
        _, x, _, grid = self.AS(x)
        AS_offset = self.AS.get_offset(grid, cell_encode=False)
        _, _, h, w = ori_x.shape
        AS_offset = torch.cat([
            AS_offset[:, 1:2, :, :],
            AS_offset[:, 0:1, :, :],
                               ], dim=1) # xy to yx
        AS_offset[:, 0] *= (0.5 * h)
        AS_offset[:, 1] *= (0.5 * w)
        """
        # DCNv2-pytorch_1.9/src/cpu/dcn_v2_im2col_cpu.cpp（参考实现）
        # const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        # // offset的具体位置
        # const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        # const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        # const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        # const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        """
        # print(self.conv_offset.dilation)
        # print(self.conv_mask.dilation)
        # print(self.AS_offset.dilation)
        # offset_dilation = self.conv_offset_dilation(x).sigmoid()
        if self.kernel_filter_wise:
            global_feat = self.global_fc(x)
            kernel_att = self.sigmoid(self.kernel_wise(global_feat))
            filter_att = self.sigmoid(self.filter_wise(global_feat))
            x = x * kernel_att

        if self.only_center:
            offset = self.conv_offset(x)
            b, c, h, w = offset.shape
            offset = offset.reshape(b, self.deform_groups, -1, h, w)
            # offset_dilation = offset_dilation.reshape(b, self.deform_groups, -1, h, w)
            # offset = offset * offset_dilation
            # print('offset', offset.shape)
            # o1, o2 = torch.chunk(offset, 2, dim=2)

            # AS_offset = self.AS_offset(x)
            # AS_offset = AS_offset.reshape(b, 1, -1, h, w).repeat(1, self.deform_groups, 1, 1, 1).contiguous()
            AS_offset = F.interpolate(AS_offset, size=(h, w), mode='bilinear')
            AS_offset = AS_offset.reshape(b, self.deform_groups, -1, h, w)
            # print('AS_offset', AS_offset.shape)
            # AS_o1, AS_o2 = torch.chunk(AS_offset, 2, dim=2)
            if self.plus_center: 
                # print('plus_center')
                offset += AS_offset.repeat(1, 1, c // self.deform_groups // 2 , 1, 1)

            o_n = self.offset_size * self.offset_size - 1
            # o1_list = list(torch.chunk(o1, o_n, dim=2))
            # o2_list = list(torch.chunk(o2, o_n, dim=2))
            # print(o2_list[0].shape, AS_o1.shape)
            offset_list = list(torch.chunk(offset, o_n, dim=2))

            # o1 = torch.cat(o1_list[:o_n // 2] + [AS_o1] + o1_list[o_n // 2:], 1)
            # o1 = torch.cat(o2_list[:o_n // 2] + [AS_o2] + o2_list[o_n // 2:], 1)
            # o1 = torch.cat([*o1_list[:o_n // 2], AS_o1, *o1_list[o_n // 2:]], 2)
            # o2 = torch.cat([*o2_list[:o_n // 2], AS_o2, *o2_list[o_n // 2:]], 2)
            # offset = torch.cat((o1, o2), dim=2).reshape(b, -1, h, w)
            # offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[o_n // 2:]], 2).reshape(b, -1, h, w)
            if self.use_offset_interp:
                offset = torch.stack([*offset_list[:o_n // 2], AS_offset, *offset_list[-(o_n // 2):]], 2) # B, DeformGroup, offset_size**2, 2, H, W
                offset = offset.permute(0, 1, 4, 5, 3, 2) # B, DeformGroup, H, W, 2, offset_size**2
                # print('offset', offset.shape)
                offset = offset.reshape(-1, 2, self.offset_size, self.offset_size)
                offset = F.interpolate(offset, size=(self.kernel_size[0], self.kernel_size[1]), mode='bilinear', align_corners = (1 == (self.offset_size // 2)))
                offset = offset.reshape(b, self.deform_groups, h, w, 2, self.kernel_size[0], self.kernel_size[1]) # b, self.deform_groups, h, w, 2, self.kernel_size[0], self.kernel_size[1]
                offset = offset.permute(0, 1, 5, 6, 4, 2, 3) # b, self.deform_groups,  self.kernel_size[0], self.kernel_size[1], 2, h, w
                offset = offset.reshape(b, -1, h, w)
            else:
                offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[-(o_n // 2):]], 2) # B, DeformGroup, offset_size * 2, H, W
                offset = offset.reshape(b, -1, h, w)
        else:
            offset = self.AS_offset(x)

        # print(offset.shape)
        # print(x.shape)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        # offset.fill_(0.0)
        # mask.fill_(1.0)
        if self.use_offset_interp:
            mask = mask.reshape(b, self.deform_groups, self.offset_size, self.offset_size, h, w)
            mask = mask.permute(0, 1, 4, 5, 2, 3) # b, deformgroup, h, w, o, o
            mask = mask.reshape(b, -1, self.offset_size, self.offset_size)
            mask = F.interpolate(mask, size=(self.kernel_size[0], self.kernel_size[1]), mode='bilinear', align_corners = (1 == (self.offset_size // 2)))
            mask = mask.reshape(b, self.deform_groups, h, w, self.kernel_size[0], self.kernel_size[1])
            mask = mask.permute(0, 1, 4, 5, 2, 3)# b, o, o, h, w,
            mask = mask.reshape(b, -1, h, w)
        # print(mask)
        # if test_offset is not None: 
        #     offset = test_offset.float()
        #     mask.fill_(1.)
        # return dcn_v2_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deform_groups)
        if self.fp16:
            x = modulated_deform_conv2d(x.float(), offset.float(), mask.float(), self.weight.float(), self.bias.float(),
                                        self.stride, self.padding,
                                        self.dilation, self.groups,
                                        self.deform_groups)
        else:
            x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                           self.stride, self.padding,
                                           self.dilation, self.groups,
                                           self.deform_groups)
        if self.kernel_filter_wise:
            x = x * filter_att
        return x



from mmcv.cnn.bricks import HSigmoid

@CONV_LAYERS.register_module('DCN_AS_DFDC')
class DCNv2_AS_DyFreqDeComposed(DCNv2_AS):
    def __init__(self, *args, only_center=True, radius=3, **kwargs):
        super().__init__(*args, only_center=only_center, radius=radius, **kwargs)
        self.lowfreq = nn.Conv2d(
                self.in_channels,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
        self.avgpool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.lowfreq.weight.data.zero_()
        self.lowfreq.bias.data.zero_()

    def forward(self, x, test_offset=None):
        if self.only_center:
            offset = self.conv_offset(x)
            b, c, h, w = offset.shape
            offset = offset.reshape(b, self.deform_groups, -1, h, w)
            # o1, o2 = torch.chunk(offset, 2, dim=2)

            AS_offset = self.AS_offset(x)
            # AS_offset = AS_offset.reshape(b, 1, -1, h, w).repeat(1, self.deform_groups, 1, 1, 1).contiguous()
            AS_offset = AS_offset.reshape(b, self.deform_groups, -1, h, w)
            # AS_o1, AS_o2 = torch.chunk(AS_offset, 2, dim=2)

            o_n = self.kernel_size[0] * self.kernel_size[1] - 1
            # o1_list = list(torch.chunk(o1, o_n, dim=2))
            # o2_list = list(torch.chunk(o2, o_n, dim=2))
            # print(o2_list[0].shape, AS_o1.shape)
            offset_list = list(torch.chunk(offset, o_n, dim=2))

            # o1 = torch.cat(o1_list[:o_n // 2] + [AS_o1] + o1_list[o_n // 2:], 1)
            # o1 = torch.cat(o2_list[:o_n // 2] + [AS_o2] + o2_list[o_n // 2:], 1)
            # o1 = torch.cat([*o1_list[:o_n // 2], AS_o1, *o1_list[o_n // 2:]], 2)
            # o2 = torch.cat([*o2_list[:o_n // 2], AS_o2, *o2_list[o_n // 2:]], 2)
            # offset = torch.cat((o1, o2), dim=2).reshape(b, -1, h, w)
            offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[o_n // 2:]], 2).reshape(b, -1, h, w)
        else:
            offset = self.AS_offset(x)

        # print(offset.shape)
        # print(x.shape)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        # print(mask)
        # if test_offset is not None: 
        #     offset = test_offset.float()
        #     mask.fill_(1.)
        # return dcn_v2_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deform_groups)
        # mask.fill_(1.)
        # offset.fill_(0.)
        # mask = mask.detach()
        # offset = offset.detach()

        low_att = self.lowfreq(x).sigmoid()
        center_idx = self.kernel_size[0] * self.kernel_size[1] - 1
        center_offset = offset.reshape(b, self.deform_groups, -1, h, w)[:, :, 2 * center_idx:2 * center_idx+2, :, :].reshape(b, -1, h, w)
        center_mask = mask.reshape(b, self.deform_groups, -1, h, w)[:, :, center_idx:center_idx+1, :, :].reshape(b, -1, h, w)
        # low_att.
        # print('low_att', low_att.mean())
        # low_att = 0.5
        low_out = 2 * low_att * modulated_deform_conv2d(self.avgpool(x), center_offset / self.stride[0], center_mask, self.weight.sum(dim=(-1, -2), keepdim=True), self.bias,
                                    1, 0,
                                    # stride=1, padding=0,
                                    self.dilation, self.groups,
                                    self.deform_groups)
        high_out = 2 * (1.0 - low_att) *  modulated_deform_conv2d(x, offset, mask, self.weight - self.weight.mean(dim=(-1, -2), keepdim=True), self.bias,
                                    self.stride, self.padding,
                                    self.dilation, self.groups,
                                    self.deform_groups)
        # For test
        # ori_out = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                             self.stride, self.padding,
        #                             self.dilation, self.groups,
        #                             self.deform_groups)
        out = low_out + high_out
        # print('diff', (ori_out - out).mean())
        return out

@CONV_LAYERS.register_module('DFDC_Conv2D')
class DyFreqDeComposedConv2D(nn.Conv2d):
    def __init__(self, *args, deformable_groups=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lowfreq = nn.Conv2d(
                self.in_channels,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
        self.highfreq = nn.Conv2d(
                self.in_channels,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
        # print(self.kernel_size)
        # print(self.dilation)
        pool_size = ((self.kernel_size[0] // 2) * self.dilation[0] * 2 + 1, (self.kernel_size[1] // 2) * self.dilation[1] * 2 + 1)
        # print(pool_size)
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size, stride=self.stride, padding=self.padding)
        c_out, c_in, kh, kw = self.weight.shape
        self.avg_weight = torch.ones(c_out, 1, kh, kw) / (self.kernel_size[0] * self.kernel_size[1])
        self.lowfreq.weight.data.zero_()
        self.lowfreq.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        self.avg_weight = self.avg_weight.to(self.weight.device) 
        low_att = self.lowfreq(input).sigmoid()
        high_att = self.highfreq(input).sigmoid()
        # print('low_att', low_att.mean())
        # low_att = 0.5
        low_out = 2 * low_att * F.conv2d(
            # self.avgpool(input), 
            F.conv2d(input, self.avg_weight, self.bias, self.stride, self.padding, self.dilation, input.size(1)),
            self.weight.sum(dim=(-1, -2), keepdim=True), self.bias, 1, 0, self.dilation, self.groups)
        # self._conv_forward(self.avgpool(input), self.weight.sum(dim=(-1, -2), keepdim=True), self.bias)
        # high_out = 2 * (1.0 - low_att) *  F.conv2d(input, self.weight - self.weight.mean(dim=(-1, -2), keepdim=True), self.bias, self.stride, self.padding, self.dilation, self.groups)
        high_out = 2 * high_att *  F.conv2d(input, self.weight - self.weight.mean(dim=(-1, -2), keepdim=True), self.bias, self.stride, self.padding, self.dilation, self.groups)
        # self._conv_forward(input, self.weight - self.weight.mean(dim=(-1, -2), keepdim=True), self.bias)
        out = low_out + high_out
        # ori_out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print((ori_out - out).mean())
        # return high_out
        return out

# https://github.com/OSVAI/ODConv/blob/main/modules/odconv.py
@CONV_LAYERS.register_module('ODFDC_Conv2D')
class OmniDyFreqDeComposedConv2D(nn.Conv2d):
    def __init__(self, *args, deformable_groups=1, cr=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.lowfreq = nn.Conv2d(
                self.in_channels,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
        # print(self.kernel_size)
        # print(self.dilation)
        pool_size = ((self.kernel_size[0] // 2) * self.dilation[0] * 2 + 1, (self.kernel_size[1] // 2) * self.dilation[1] * 2 + 1)
        # print(pool_size)
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size, stride=self.stride, padding=self.padding)
        c_out, c_in, kh, kw = self.weight.shape
        self.kernel_num = self.kernel_size[0] * self.kernel_size[1]
        self.avg_weight = torch.ones(c_out, 1, kh, kw) / self.kernel_num
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_channels, self.in_channels // cr, kernel_size=1, stride=self.stride, padding=0, dilation=self.dilation, bias=True),
            nn.SyncBatchNorm(self.in_channels // cr),
            # nn.BatchNorm2d(attention_channel),
            nn.ReLU(True)
        )
        self.kernel_wise = nn.Conv2d(self.in_channels // cr, self.in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.filter_wise = nn.Conv2d(self.in_channels // cr, self.in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.lowpart = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        # self.sigmoid = HSigmoid()
        self.sigmoid = nn.Sigmoid()
        
        self.lowfreq.weight.data.zero_()
        self.lowfreq.bias.data.zero_()
        self.kernel_wise.weight.data.zero_()
        self.kernel_wise.bias.data.zero_()
        self.filter_wise.weight.data.zero_()
        self.filter_wise.bias.data.zero_()
        
        self.lowpart.weight.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        self.avg_weight = self.avg_weight.to(self.weight.device) 
        low_att = self.sigmoid(self.lowfreq(input))
        global_res = self.global_fc(input)
        kernel_att = self.sigmoid(self.kernel_wise(global_res))
        filter_att = self.sigmoid(self.filter_wise(global_res))

        # print('kernel_att', kernel_att.shape)
        # print('filter_att', filter_att.shape)
        # print('low_att', low_att.mean())
        # low_att = 0.5
        # kernel_att = 0.5
        # filter_att = 0.5
        low_out = 2 * low_att * F.conv2d(
            # self.avgpool(input), 
            F.conv2d(2 * kernel_att * input, self.avg_weight, self.bias, self.stride, self.padding, self.dilation, input.size(1)),
            self.weight.sum(dim=(-1, -2), keepdim=True) + self.lowpart.weight, self.bias, 1, 0, self.dilation, self.groups)
        # self._conv_forward(self.avgpool(input), self.weight.sum(dim=(-1, -2), keepdim=True), self.bias)
        high_out = 2 * (1.0 - low_att) *  F.conv2d(
            2 * (1.0 - kernel_att) * input, 
            self.weight - self.weight.mean(dim=(-1, -2), keepdim=True) - self.lowpart.weight / self.kernel_num, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # self._conv_forward(input, self.weight - self.weight.mean(dim=(-1, -2), keepdim=True), self.bias)
        out = 2 * filter_att * low_out + 2 * (1.0 - filter_att) * high_out
        # ori_out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print((ori_out - out).mean())
        # return high_out
        return out

class ModulatedDeformConv2dPackFP16(ModulatedDeformConv2dPack):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x.float(), offset.float(), mask.float(), self.weight.float(), self.bias.float(),
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
    
@CONV_LAYERS.register_module('FreqDecomp_DCNv2')
class FreqDecompDCNv2(ModulatedDeformConv2dPack):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    def __init__(self, 
                 *args, 
                # k_list=[11, 9, 7, 5, 3],
                k_list=[3, 5, 7, 9, 11],
                # freq_list=[2, 3, 5, 7, 9, 11],
                fs_feat='feat',
                lp_type='freq',
                act='sigmoid',
                channel_group=1,
                channel_bn=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        if fs_feat == 'offset_conv1':
            self.freq_weight_conv = nn.Conv2d(in_channels=self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 
                                            out_channels=len(k_list) + 1, 
                                            stride=1,
                                            kernel_size=1, padding=0, bias=True)
        elif fs_feat == 'offset_conv3':
            self.freq_weight_conv = nn.Conv2d(in_channels=self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 
                                            out_channels=len(k_list) + 1, 
                                            stride=1,
                                            kernel_size=3, padding=1, bias=True)
        elif fs_feat == 'feat':
            self.freq_weight_conv = nn.Conv2d(in_channels=self.in_channels, 
                                            out_channels=len(k_list) + 1, 
                                            stride=1,
                                            kernel_size=3, padding=1, bias=True)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReflectionPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type == 'freq_channel_att':
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, bias=0, groups=channel_group),
                nn.SyncBatchNorm(self.in_channels) if channel_bn else nn.Identity(),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.freq_weight_conv.weight.data.zero_()
        self.freq_weight_conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        # freq_weight = self.freq_weight_conv(out)
        # freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        # freq_weight = self.freq_weight_conv(out).sigmoid()
        if self.fs_feat in ('offset_conv1', 'offset_conv3'):
            up_out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=(x.shape[-1] // 2) == 1)
            freq_weight = self.freq_weight_conv(up_out)
        elif self.fs_feat in ('feat',):
            freq_weight = self.freq_weight_conv(x)
        
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        
        x_list = []
        if self.lp_type == 'avgpool':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(freq_weight[:, idx:idx+1] * high_part)
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1])
        elif self.lp_type == 'freq':
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(freq_weight[:, idx:idx+1] * high_part)
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1])
        elif self.lp_type == 'freq_channel_att':
            pre_freq = 1
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            h, w =  int(h), int(w) 
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                channel_att_mask = mask.clone()
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part

                channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                channel_att_mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                pre_freq = int(freq)
                c_att = self.channel_att((x_fft * channel_att_mask).abs())
                
                x_list.append(freq_weight[:, idx:idx+1] * high_part * (c_att + 1))

            channel_att_mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            c_att = self.channel_att((x_fft * channel_att_mask).abs())
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1] * (c_att + 1))
        elif self.lp_type == 'freq_channel_att_dev':
            pre_freq = 1
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x, device=x.device)
                mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                pre_freq = freq
                # if idx == len(self.k_list) - 1:
                #     pass
                # else:
                #     freq_next = self.k_list[idx + 1]
                #     mask[:,:,int(h/2 - h/(2 * freq_next)):int(h/2 + h/(2 * freq_next)), int(w/2 - w/(2 * freq_next)):int(w/2 + w/(2 * freq_next))] = 1
                x_fft_masked = x_fft * mask
                c_att = self.channel_att(x_fft_masked.abs())
                band_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft_masked)).real
                x_list.append(freq_weight[:, idx:idx+1] * band_part * (c_att + 1))

            mask = torch.zeros_like(x, device=x.device)
            mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            x_fft_masked = x_fft * mask
            c_att = self.channel_att(x_fft_masked.abs())
            band_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft_masked)).real
            x_list.append(pre_x * freq_weight[:, len(x_list):len(x_list)+1] * (c_att + 1))
        
        x = sum(x_list)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class SimAT(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super().__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
    
@CONV_LAYERS.register_module('FreqDecomp2_DCNv2')
class FreqDecomp2DCNv2(ModulatedDeformConv2dPack):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    def __init__(self, 
                 *args, 
                # k_list=[11, 9, 7, 5, 3],
                k_list=[3, 5, 7, 9, 11],
                # freq_list=[2, 3, 5, 7, 9, 11],
                att_type='simat',
                lp_type='freq',
                act='sigmoid',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.att_type = att_type
        self.lp_type = lp_type
        
        if self.att_type == 'simat':
            self.att = SimAT()
        elif self.att_type == 'freq':
            pass
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReflectionPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'freq':
            pass
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        # self.freq_weight_conv.weight.data.zero_()
        # self.freq_weight_conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        # freq_weight = self.freq_weight_conv(out)
        # freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        # freq_weight = self.freq_weight_conv(out).sigmoid()
        
        x_list = []
        if self.lp_type == 'avgpool':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(high_part)
            x_list.append(pre_x)
        elif self.lp_type == 'freq':
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:,:,int(h/2 - h/(2 * freq)):int(h/2 + h/(2 * freq)), int(w/2 - w/(2 * freq)):int(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(high_part)
            x_list.append(pre_x)

        x_list = multi_apply(self.att, x_list)

        x = sum(x_list)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

import mmcv
from mmcv.cnn import ConvModule
class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                        #   dict(type='HSigmoid', bias=3.0, divisor=6.0),
                          dict(type='Sigmoid')
                          )
                          ):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            # out_channels=make_divisible(channels // ratio, 8),
            out_channels=channels // ratio,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            # in_channels=make_divisible(channels // ratio, 8),
            in_channels=channels // ratio,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

import torch.nn.functional as F
def generate_laplacian_pyramid(input_tensor, num_levels, size_align=True, mode='bilinear'):
    pyramid = []
    current_tensor = input_tensor
    _, _, H, W = current_tensor.shape
    for _ in range(num_levels):
        b, _, h, w = current_tensor.shape
        downsampled_tensor = F.interpolate(current_tensor, (h//2 + h%2, w//2 + w%2), mode=mode, align_corners=(H%2) == 1) # antialias=True
        if size_align: 
            # upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode='bilinear', align_corners=(H%2) == 1)
            # laplacian = current_tensor - upsampled_tensor
            # laplacian = F.interpolate(laplacian, (H, W), mode='bilinear', align_corners=(H%2) == 1)
            upsampled_tensor = F.interpolate(downsampled_tensor, (H, W), mode=mode, align_corners=(H%2) == 1)
            laplacian = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H%2) == 1) - upsampled_tensor
            # print(laplacian.shape)
        else:
            upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode=mode, align_corners=(H%2) == 1)
            laplacian = current_tensor - upsampled_tensor
        pyramid.append(laplacian)
        current_tensor = downsampled_tensor
    if size_align: current_tensor = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H%2) == 1)
    pyramid.append(current_tensor)
    return pyramid
                
class FrequencySelection(nn.Module):
    def __init__(self, 
                in_channels,
                k_list=[2],
                # freq_list=[2, 3, 5, 7, 9, 11],
                lowfreq_att=True,
                fs_feat='feat',
                lp_type='freq_channel_att',
                act='sigmoid',
                channel_res=True,
                # residual=False,
                spatial='conv',
                spatial_group=1,
                spatial_kernel=3,
                init='zero',
                global_selection=False,
                ):
        super().__init__()
        # k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        self.channel_res = channel_res
        # self.residual = residual
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=spatial_kernel, 
                                            groups=self.spatial_group,
                                            padding=spatial_kernel//2, 
                                            bias=True)
                if init == 'zero':
                    freq_weight_conv.weight.data.zero_()
                    freq_weight_conv.bias.data.zero_()   
                else:
                    # raise NotImplementedError
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)
        elif spatial == 'cbam': 
            self.freq_weight_conv = SpatialGate(out=len(k_list) + 1)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReplicationPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type in ('freq_channel_att', 'freq_channel_att_reduce_high'):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            self.channel_att[1].weight.data.zero_()
            self.channel_att[1].bias.data.zero_()
            # self.channel_att.weight.data.zero_()
        elif self.lp_type in ('freq_eca', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.ModuleList(
                [eca_layer(self.in_channels, k_size=9) for _ in range(len(k_list) + 1)]
            )
        elif self.lp_type in ('freq_channel_se', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = SELayer(self.in_channels, ratio=16)
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=1, 
                                            groups=self.spatial_group,
                                            padding=0, 
                                            bias=True)
            self.global_selection_conv_imag = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=1, 
                                            groups=self.spatial_group,
                                            padding=0, 
                                            bias=True)
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()  
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()  

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        att_feat:feat for gen att
        """
        # freq_weight = self.freq_weight_conv(x)
        # self.sp_act(freq_weight)
        # if self.residual: x_residual = x.clone()
        if att_feat is None: att_feat = x
        x_list = []
        if self.lp_type == 'avgpool':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            pre_x = x
            b, _, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                # x_list.append(freq_weight[:, idx:idx+1] * high_part)
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre_x)
        elif self.lp_type == 'laplacian':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            # pre_x = x
            b, _, h, w = x.shape
            pyramids = generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            # print('pyramids', len(pyramids))
            for idx, avg in enumerate(self.k_list):
                # print(idx)
                high_part = pyramids[idx]
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pyramids[-1].reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pyramids[-1])
        elif self.lp_type == 'freq':
            pre_x = x.clone()
            b, _, h, w = x.shape
            # b, _c, h, w = freq_weight.shape
            # freq_weight = freq_weight.reshape(b, self.spatial_group, -1, h, w)
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
            if self.global_selection:
                # global_att_real = self.global_selection_conv_real(x_fft.real)
                # global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # global_att_imag = self.global_selection_conv_imag(x_fft.imag)
                # global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # x_fft = x_fft.reshape(b, self.spatial_group, -1, h, w)
                # x_fft.real *= global_att_real
                # x_fft.imag *= global_att_imag
                # x_fft = x_fft.reshape(b, -1, h, w)
                # 将x_fft复数拆分成实部和虚部
                x_real = x_fft.real
                x_imag = x_fft.imag
                # 计算实部的全局注意力
                global_att_real = self.global_selection_conv_real(x_real)
                global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # 计算虚部的全局注意力
                global_att_imag = self.global_selection_conv_imag(x_imag)
                global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # 重塑x_fft为形状为(b, self.spatial_group, -1, h, w)的张量
                x_real = x_real.reshape(b, self.spatial_group, -1, h, w)
                x_imag = x_imag.reshape(b, self.spatial_group, -1, h, w)
                # 分别应用实部和虚部的全局注意力
                x_fft_real_updated = x_real * global_att_real
                x_fft_imag_updated = x_imag * global_att_imag
                # 合并为复数
                x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
                # 重塑x_fft为形状为(b, -1, h, w)的张量
                x_fft = x_fft_updated.reshape(b, -1, h, w)

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](x)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](x)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre_x)
        elif self.lp_type in ('freq_channel_att', 'freq_eca', 'freq_channel_se'):
            b, _, h, w = x.shape
            pre_freq = 1
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            h, w =  int(h), int(w) 
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                channel_att_mask = mask.clone()
                mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part
                # print('hw:', h, w)
                # print(idx, ':', freq, round(h/2 - h/(2 * freq)), round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)), round(w/2 + w/(2 * freq)))
                channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                channel_att_mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                pre_freq = int(freq)
                freq_weight = self.freq_weight_conv_list[idx](x)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = tmp.reshape(b, -1, h, w)
                if isinstance(self.channel_att, nn.ModuleList):
                    c_att = self.channel_att[idx]((x_fft * channel_att_mask).abs())
                else:
                    c_att = self.channel_att((x_fft * channel_att_mask).abs())
                    # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
                c_att = (c_att + 0.5) if self.channel_res else (c_att * 2.0)
                x_list.append(tmp * high_part * c_att)

            channel_att_mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            if isinstance(self.channel_att, nn.ModuleList):
                c_att = self.channel_att[len(x_list)]((x_fft * channel_att_mask).abs())
            else:
                c_att = self.channel_att((x_fft * channel_att_mask).abs())
                # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
            c_att = (c_att + 0.5) if self.channel_res else (c_att * 2.0)
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](x)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = tmp.reshape(b, -1, h, w)
                x_list.append(tmp * c_att)
            else:
                x_list.append(pre_x * c_att)
                # for i in x_list: print(i.shape)
        elif self.lp_type == 'freq_channel_att_reduce_high':
            pre_freq = 1
            pre_x = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x))
            h, w = x.shape[-2:]
            h, w =  int(h), int(w) 
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                channel_att_mask = mask.clone()
                mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask)).real
                high_part = pre_x - low_part
                pre_x = low_part

                channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
                channel_att_mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 0.0
                pre_freq = int(freq)
                # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
                c_att = self.channel_att((x_fft * channel_att_mask).abs())
                x_list.append((1. - freq_weight[:, idx:idx+1]) * high_part * (1. - c_att))
            channel_att_mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            channel_att_mask[:,:,round(h/2 - h/(2 * pre_freq)):round(h/2 + h/(2 * pre_freq)), round(w/2 - w/(2 * pre_freq)):round(w/2 + w/(2 * pre_freq))] = 1.0
            # c_att = self.channel_att((x_fft * channel_att_mask).abs() / (F.adaptive_avg_pool2d(x_fft.abs(), 1) + 1e-8))
            c_att = self.channel_att((x_fft * channel_att_mask).abs())
            x_list.append(pre_x * (freq_weight[:, len(x_list):len(x_list)+1] + 1) * (c_att + 1))
        x = sum(x_list)
        # if self.residual: x += x_residual
        return x

class FrequencyNorm(nn.Module):
    def __init__(self, in_channels=1, pooling_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_size = pooling_size
        k_tl = [
            [1/3, 1/3],
            [1/3, -1],
        ]
        k_bl = [
            [1/3, -1],
            [1/3, 1/3],
        ]
        k_tr = [
            [1/3, 1/3],
            [-1,  1/3],
        ]
        k_br = [
            [-1,  1/3],
            [1/3, 1/3],
        ]
        self.register_buffer('k_tl', torch.Tensor(k_tl)[None, None,])#.repeat(in_channels, in_channels, 1, 1)) # c, c, k, k
        self.register_buffer('k_bl', torch.Tensor(k_bl)[None, None,])#.repeat(in_channels, in_channels, 1, 1)) # c, c, k, k
        self.register_buffer('k_tr', torch.Tensor(k_tr)[None, None,])#.repeat(in_channels, in_channels, 1, 1)) # c, c, k, k
        self.register_buffer('k_br', torch.Tensor(k_br)[None, None,])#.repeat(in_channels, in_channels, 1, 1)) # c, c, k, k

    def forward(self, x, pooling_size=None):
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho')).abs()
        # x_fft = F.avg_pool2d(x_fft, kernel_size=5, padding=2, stride=1)
        # x_fft = F.avg_pool2d(x_fft, kernel_size=3, padding=1, stride=1)
        # x_fft = F.adaptive_avg_pool2d(x_fft, output_size=(32, 32))
        if pooling_size is None:pooling_size = self.pooling_size
        x_fft = F.adaptive_avg_pool2d(x_fft, output_size=pooling_size)
        b, c, h, w = x_fft.shape
        # x_fft[:, :, h//2:] = x_fft[:, :, h//2::-1]
        # x_fft[:, :, :, w//2:] = x_fft[:, :, :, w//2::-1]
        tl = F.conv2d(x_fft[:, :, :h//2, :w//2], self.k_tl.repeat(c, 1, 1, 1), padding=0, groups=c, stride=1).reshape(b, c, -1)
        bl = F.conv2d(x_fft[:, :, h//2:, :w//2], self.k_bl.repeat(c, 1, 1, 1), padding=0, groups=c, stride=1).reshape(b, c, -1)
        tr = F.conv2d(x_fft[:, :, :h//2, w//2:], self.k_tr.repeat(c, 1, 1, 1), padding=0, groups=c, stride=1).reshape(b, c, -1)
        br = F.conv2d(x_fft[:, :, h//2:, w//2:], self.k_br.repeat(c, 1, 1, 1), padding=0, groups=c, stride=1).reshape(b, c, -1)
        # print(tl.clamp_min_(0).mean(-1))
        # print(bl.clamp_min_(0).mean(-1))
        # print(tr.clamp_min_(0).mean(-1))
        # print(br.clamp_min_(0).mean(-1))
        res =  torch.cat([tl, bl, tr, br], dim=-1)
        res.clamp_min_(0)
        return res.mean(-1)

class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
    
class MultiFreqBandConv(nn.Module):
    def __init__(self, in_channels, out_channel, freq_band=4, kernel_size=1, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.freq_band = freq_band
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(freq_band):
            freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                        out_channels=out_channel, 
                                        stride=1,
                                        kernel_size=kernel_size, 
                                        groups=1,
                                        padding=kernel_size//2, 
                                        bias=True)
            freq_weight_conv.weight.data.zero_()
            freq_weight_conv.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] / freq_band)   
            self.freq_weight_conv_list.append(freq_weight_conv)
    def forward(self, x):
        pyramids = generate_laplacian_pyramid(x, self.freq_band, size_align=True)
        res = 0
        for i in range(self.freq_band):
            res += self.freq_weight_conv_list[i](pyramids[i])
        return res


import torch.fft as fft
from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.runner import force_fp32
@CONV_LAYERS.register_module('AdaDilatedConv')
class AdaptiveDilatedConv(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2
    def __init__(self, *args, 
                 offset_freq=None,
                 padding_mode=None,
                 kernel_decompose=None,
                 conv_type='conv',
                 sp_att=False,
                 pre_fs=True, # False, use dilation
                 epsilon=0,
                 use_zero_dilation=False,
                 fs_cfg={
                    'k_list':[3,5,7,9],
                    'fs_feat':'feat',
                    # 'lp_type':'freq_eca',
                    # 'lp_type':'freq_channel_att',
                    # 'lp_type':'freq',
                    'lp_type':'avgpool',
                    # 'lp_type':'laplacian',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'channel_res':True,
                    'spatial_group':1,
                },
                 **kwargs):
        super().__init__(*args, **kwargs)
        if padding_mode == 'zero':
            self.PAD = nn.ZeroPad2d(self.kernel_size[0]//2)
        elif padding_mode == 'repeat':
            self.PAD = nn.ReplicationPad2d(self.kernel_size[0]//2)
        else:
            self.PAD = nn.Identity()

        self.kernel_decompose = kernel_decompose
        if kernel_decompose == 'both':
            self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
            self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'high':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'low':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        self.conv_type = conv_type
        if conv_type == 'conv':
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
                dilation=1,
                bias=True)
        elif conv_type == 'multifreqband':
            self.conv_offset = MultiFreqBandConv(self.in_channels, self.deform_groups * 1, freq_band=4, kernel_size=1, dilation=self.dilation)
        else:
            raise NotImplementedError
            pass
        # self.conv_offset_low = nn.Sequential(
        #     nn.AvgPool2d(
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         padding=1,
        #     ),
        #     nn.Conv2d(
        #         self.in_channels,
        #         self.deform_groups * 1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False),
        # )

        # self.conv_offset_high = nn.Sequential(
        #     LHPFConv3(channels=self.in_channels, stride=1, padding=1, residual=False),
        #     nn.Conv2d(
        #         self.in_channels,
        #         self.deform_groups * 1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         dilation=1,
        #         bias=True),
        # )
        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
            dilation=1,
            bias=True)
        if sp_att:
            self.conv_mask_mean_level = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
                dilation=1,
                bias=True)
        
        self.offset_freq = offset_freq

        if self.offset_freq in ('FLC_high', 'FLC_res'):
            self.LP = FLC_Pooling(freq_thres=min(0.5 * 1 / self.dilation[0], 0.25))
        elif self.offset_freq in ('SLP_high', 'SLP_res'):
            self.LP = StaticLP(self.in_channels, kernel_size=3, stride=1, padding=1, alpha=8)
        elif self.offset_freq is None:
            pass
        else:
            raise NotImplementedError

        # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
        offset = [-1, -1,  -1, 0,   -1, 1,
                  0, -1,   0, 0,    0, 1,
                  1, -1,   1, 0,    1,1]
        offset = torch.Tensor(offset)
        # offset[0::2] *= self.dilation[0]
        # offset[1::2] *= self.dilation[1]
        # a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
        self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None])) # B, G, 18, 1, 1
        if fs_cfg is not None:
            if pre_fs:
                self.FS = FrequencySelection(self.in_channels, **fs_cfg)
            else:
                self.FS = FrequencySelection(1, **fs_cfg) # use dilation
        self.pre_fs = pre_fs
        self.epsilon = epsilon
        self.use_zero_dilation = use_zero_dilation
        self.init_weights()

    def freq_select(self, x):
        if self.offset_freq is None:
            res = x
        elif self.offset_freq in ('FLC_high', 'SLP_high'):
            res = x - self.LP(x)
        elif self.offset_freq in ('FLC_res', 'SLP_res'):
            res = 2 * x - self.LP(x)
        else:
            raise NotImplementedError
        return res

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            # if isinstanace(self.conv_offset, nn.Conv2d):
            if self.conv_type == 'conv':
                self.conv_offset.weight.data.zero_()
                # self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + 1e-4)
                self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + self.epsilon)
            # self.conv_offset.bias.data.zero_()
        # if hasattr(self, 'conv_offset'):
            # self.conv_offset_low[1].weight.data.zero_()
        # if hasattr(self, 'conv_offset_high'):
            # self.conv_offset_high[1].weight.data.zero_()
            # self.conv_offset_high[1].bias.data.zero_()
        if hasattr(self, 'conv_mask'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

        if hasattr(self, 'conv_mask_mean_level'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    # @force_fp32(apply_to=('x',))
    # @force_fp32
    def forward(self, x):
        # offset = self.conv_offset(self.freq_select(x)) + self.conv_offset_low(self.freq_select(x))
        if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            c_att1, f_att1, _, _, = self.OMNI_ATT1(x)
            c_att2, f_att2, _, _, = self.OMNI_ATT2(x)
        elif hasattr(self, 'OMNI_ATT'):
            c_att, f_att, _, _, = self.OMNI_ATT(x)
        
        if self.conv_type == 'conv':
            offset = self.conv_offset(self.PAD(self.freq_select(x)))
        elif self.conv_type == 'multifreqband':
            offset = self.conv_offset(self.freq_select(x))
        # high_gate = self.conv_offset_high(x)
        # high_gate = torch.exp(-0.5 * high_gate ** 2)
        # offset = F.relu(offset, inplace=True) * self.dilation[0] - 1 # ensure > 0
        if self.use_zero_dilation:
            raise NotImplementedError
            offset = (F.relu(offset + 1, inplace=True) - 1) * self.dilation[0] # ensure > 0
        else:
            # offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
            offset = offset.abs() * self.dilation[0] # ensure > 0
            # offset[offset<0] = offset[offset<0].exp() - 1
        # print(offset.mean(), offset.std(), offset.max(), offset.min())
        if hasattr(self, 'FS') and (self.pre_fs==False): x = self.FS(x, F.interpolate(offset, x.shape[-2:], mode='bilinear', align_corners=(x.shape[-1]%2) == 1))
        # print(offset.max(), offset.abs().min(), offset.abs().mean())
        # offset *= high_gate # ensure > 0
        b, _, h, w = offset.shape
        offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
        # offset = offset.reshape(b, self.deform_groups, -1, h, w).repeat(1, 1, 9, 1, 1)
        # offset[:, :, 0::2, ] *= self.dilated_offset[:, :, 0::2, ]
        # offset[:, :, 1::2, ] *= self.dilated_offset[:, :, 1::2, ]
        offset = offset.reshape(b, -1, h, w)
        
        x = self.PAD(x)
        mask = self.conv_mask(x)
        mask = mask.sigmoid()
        # print(mask.shape)
        # mask = mask.reshape(b, self.deform_groups, -1, h, w).softmax(dim=2)
        if hasattr(self, 'conv_mask_mean_level'):
            mask_mean_level = torch.sigmoid(self.conv_mask_mean_level(x)).reshape(b, self.deform_groups, -1, h, w)
            mask = mask * mask_mean_level
        mask = mask.reshape(b, -1, h, w)
        
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, c_out, c_in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
            adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (f_att1.unsqueeze(2) * 2) + (adaptive_weight - adaptive_weight_mean) * (c_att2.unsqueeze(1) * 2) * (f_att2.unsqueeze(2) * 2)
            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                self.stride, (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD, nn.Identity) else (0, 0), #padding
                                (1, 1), # dilation
                                self.groups * b, self.deform_groups * b)
        elif hasattr(self, 'OMNI_ATT'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, c_out, c_in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
            if self.kernel_decompose == 'high':
                adaptive_weight = adaptive_weight_mean + (adaptive_weight - adaptive_weight_mean) * (c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2)
            elif self.kernel_decompose == 'low':
                adaptive_weight = adaptive_weight_mean * (c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2) + (adaptive_weight - adaptive_weight_mean) 
                
            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            # adaptive_bias = self.unsqueeze(0).repeat(b, 1, 1, 1, 1)
            # print(adaptive_weight.shape)
            # print(offset.shape)
            # print(mask.shape)
            # print(x.shape)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                        self.stride, (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD, nn.Identity) else (0, 0), #padding
                                        (1, 1), # dilation
                                        self.groups * b, self.deform_groups * b)
        else:
            x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                        self.stride, (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD, nn.Identity) else (0, 0), #padding
                                        (1, 1), # dilation
                                        self.groups, self.deform_groups)
        # x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, self.groups,
        #                                self.deform_groups)
        # if hasattr(self, 'OMNI_ATT'): x = x * f_att
        return x.reshape(b, -1, h, w)

@CONV_LAYERS.register_module('AAConv')
class AntiAliasingConv(nn.Conv2d):
    def __init__(self,
                 *args, 
                 freq_select_cfg={
                    # 'k_list':[2],
                    'fs_feat':'feat',
                    # 'lp_type':'freq_eca',
                    # 'lp_type':'freq_channel_att',
                    'lp_type':'freq',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'channel_res':True,
                    'spatial_group':1,
                },
                pre_fs=True,
                res_path='extra_conv1x1',
                anti_aliasing_path=False,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.freq_thres = 1 / self.dilation[0] * 0.5
        self.freq_select_cfg = freq_select_cfg
        self.pre_fs = pre_fs
        if freq_select_cfg is not None:
            step = self.dilation[0] // 8
            step = max(step, 1)
            if self.dilation[0] > 1:
                k_list = [self.dilation[0] / i for i in range(1, self.dilation[0], step)][::-1]
            else:
                k_list = [2]
            # _cfg = freq_select_cfg
            _cfg = {'k_list':k_list}
            _cfg.update(freq_select_cfg)
            print(_cfg)
            self.FS = FrequencySelection(in_channels=self.in_channels, **_cfg)
            self.res_path = res_path
            self.anti_aliasing_path = anti_aliasing_path
            if self.anti_aliasing_path and res_path in ('extra_conv1x1', 'high_extra_conv1x1'):
                self.extra_conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, dilation=1, padding=0, bias=self.bias)

    def forward(self, x):
        # if self.dilation == 1: return super().forward(x)
        if self.FS is not None:
            if self.pre_fs:
                x = super().forward(x)
            else:
                x = super().forward(x)
                x = self.FS(x)
        else:
            pass

        if self.dilation == 1 or (not self.anti_aliasing_path):
            # print(self.anti_aliasing_path)
            return x
        else:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))
            mask = torch.zeros_like(low_part, device=low_part.device)
            # mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
            _, _, h, w = x.shape
            mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
            low = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
            if self.res_path == 'conv1x1':
                high_res = F.conv2d(x-low, self.weight.sum(dim=(-1, -2), keepdim=True), self.bias, self.stride, 0, self.dilation, self.groups)
            elif self.res_path == 'high_extra_conv1x1':
                high_res = self.extra_conv1x1(x-low)
            elif self.res_path == 'extra_conv1x1':
                high_res = self.extra_conv1x1(x)
            elif self.res_path == 'none':
                high_res = 0
            else:
                raise NotImplementedError
            return low + high_res
    
    # def extra_repr(self):
    #     s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
    #          ', stride={stride}')
    #     if self.padding != (0,) * len(self.padding):
    #         s += ', padding={padding}'
    #     if self.dilation != (1,) * len(self.dilation):
    #         s += ', dilation={dilation}'
    #     if self.output_padding != (0,) * len(self.output_padding):
    #         s += ', output_padding={output_padding}'
    #     if self.groups != 1:
    #         s += ', groups={groups}'
    #     if self.bias is None:
    #         s += ', bias=False'
    #     if self.padding_mode != 'zeros':
    #         s += ', padding_mode={padding_mode}'
    #     s += ', res_path={self.res_path}'
    #     if self.freq_select_cfg != None:
    #         for k in self.freq_select_cfg:
    #             s += ', {k}={self.freq_select_cfg[k]}'
    #     return s.format(**self.__dict__)

@CONV_LAYERS.register_module('AAConv2')
class AntiAliasingConv2(nn.Conv2d):
    def __init__(self,
                 *args, 
                 compress_ratio=4,
                 lp_kernel=None,
                 lp_group=1,
                 att_group=1,
                 pre_filter=True,
                 locality_weight=None,
                 lp_bank=['PALP'],
                 use_BFM=False,
                 freq_select_cfg={
                    # 'k_list':[2],
                    'fs_feat':'feat',
                    # 'lp_type':'freq_eca',
                    # 'lp_type':'freq_channel_att',
                    'lp_type':'freq',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'channel_res':True,
                    'spatial_group':1,
                },
                 **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.freq_thres = 1 / self.dilation[0] * 0.5
        if lp_kernel is None: lp_kernel = 1 + 2 * self.dilation[0]
        self.lp_bank = lp_bank
        self.pre_filter = pre_filter
        self.compress_ratio = compress_ratio
        self.use_BFM = use_BFM
        self.att_group = att_group
        if self.dilation[0] > 1:
            if 'PALP' in self.lp_bank:
                self.PALP = PALowPass(
                    channels=self.in_channels, scale_factor=1,
                    up_kernel=lp_kernel,
                    up_group=lp_group,
                    encoder_kernel=3,
                    encoder_dilation=1,
                    compressed_channels=self.in_channels // compress_ratio,
                    locality_weight=locality_weight)
            if 'SLP' in self.lp_bank:
                self.SLP = StaticLP(channels=self.in_channels, kernel_size=lp_kernel, padding=lp_kernel//2, stride=1)
            if 'FS' in self.lp_bank:
                step = self.dilation[0] // 8
                step = max(step, 1)
                if self.dilation[0] > 1:
                    k_list = [self.dilation[0] / i for i in range(1, self.dilation[0], step)][::-1]
                else:
                    k_list = [2]
                # _cfg = freq_select_cfg
                _cfg = {'k_list':k_list}
                _cfg.update(freq_select_cfg)
                print(_cfg)
                self.FS = FrequencySelection(in_channels=self.in_channels, **_cfg)
        # print()
            if len(self.lp_bank) > 0:
                self.att_conv_list = nn.ModuleList()
                for i in range(len(self.lp_bank) + 1):
                    _conv = nn.Conv2d(in_channels=self.in_channels, 
                                    out_channels=self.att_group , 
                                    stride=1,
                                    kernel_size=3, 
                                    groups=self.att_group,
                                    padding=3//2, 
                                    bias=True)
                    # _conv.weight.data.zero_()
                    if i == 0: 
                        _conv.bias.data.fill_(5) 
                    else:
                        _conv.bias.data.fill_(-5) 
                    self.att_conv_list.append(_conv)
            # if init == 'zero':
                # freq_weight_conv.weight.data.zero_()
                # freq_weight_conv.bias.data.zero_()  
                # freq_weight_conv.bias.data.fill_(-5)  
                # freq_weight_conv.bias.data[0].fill_(5)
        if use_BFM:
            alpha = 1
            BFM = np.zeros((self.in_channels, 1, self.kernel_size[0], self.kernel_size[0]))
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[0]):
                    point_1 = (i, j)
                    point_2 = (self.kernel_size[0]//2, self.kernel_size[0]//2)
                    dist = distance.euclidean(point_1, point_2)
                    BFM[:, :, i, j] = alpha / (dist + alpha)
            self.register_buffer('BFM', torch.Tensor(BFM))
                
    def forward(self, x):
        if self.training and self.use_BFM:
            with torch.no_grad():
             # https://discuss.pytorch.org/t/set-nn-parameter-during-training/178163/3
                # print(self.BFM)
                # print(self.weight)
                self.weight.copy_(self.weight * self.BFM)
                
        if self.dilation[0] < 2:
            # print(self.freq_thres)
            return super().forward(x)
        else:
            if self.pre_filter:
                att_idx=0
                res = x * self.att_conv_list[att_idx](x).sigmoid()
                att_idx += 1
                if 'PALP' in self.lp_bank: 
                    res += self.PALP(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'SLP' in self.lp_bank: 
                    res += self.SLP(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'FLC' in self.lp_bank: 
                    res += self.FLC(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'FS' in self.lp_bank: 
                    res += self.FS(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                res = super().forward(res)
            else:
                x = super().forward(x)
                att_idx=0
                res = x * self.att_conv_list[att_idx](x).sigmoid()
                att_idx += 1
                if 'PALP' in self.lp_bank: 
                    res += self.PALP(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'SLP' in self.lp_bank: 
                    res += self.SLP(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'FLC' in self.lp_bank: 
                    res += self.FLC(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
                if 'FS' in self.lp_bank: 
                    res += self.FS(x) * self.att_conv_list[att_idx](x).sigmoid()
                    att_idx += 1
            return res

    def FLC(self, x):
        if self.dilation[0] < 2:
            return x
        else:
            low_part = torch.fft.fftshift(torch.fft.fft2(x))
            mask = torch.zeros_like(low_part, device=low_part.device)
            # mask[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)] = 1.0
            _, _, h, w = x.shape
            mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
            low = torch.fft.ifft2(torch.fft.ifftshift(low_part * mask)).real
            return low
        
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', FLC_thres={freq_thres}'
        s += ', lp_bank={lp_bank}'
        s += ', compress_ratio={compress_ratio}'
        s += ', pre_filter={pre_filter}'
        return s.format(**self.__dict__)  

class PALowPass(CARAFEPack):
    """A unified package of CARAFE upsampler that contains: 1) channel
    compressor 2) content encoder 3) CARAFE op.

    Official implementation of ICCV 2019 paper
    `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 *args,
                 locality_weight=None,
                 **kwargs,
                #  channels: int,
                #  scale_factor: int,
                #  up_kernel: int = 5,
                #  up_group: int = 1,
                #  encoder_kernel: int = 3,
                #  encoder_dilation: int = 1,
                #  compressed_channels: int = 64
                 ):
        super().__init__(*args, **kwargs)
        # print('Filter size [%i]'%filt_size)
        self.filt_size = kwargs['up_kernel']
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        k_weight = torch.Tensor(a[:,None]*a[None,:])
        k_weight = k_weight/torch.sum(k_weight)
        # kh, kw = filt
        self.register_buffer('k_weight', k_weight[None,None,:,:].reshape(1, 1, -1, 1, 1))
        self.locality_weight=locality_weight

    def feature_reassemble(self, x: Tensor, mask: Tensor) -> Tensor:
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x: Tensor) -> Tensor:
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x
    
    def kernel_normalizer(self, mask: Tensor) -> Tensor:
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w) 
        if self.locality_weight is not None:
            mask = mask * self.k_weight
            # print(mask.shape)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

from scipy.spatial import distance
class StaticLP(nn.Module):
    """
    Static Low Pass Filter
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, alpha=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.PAD = nn.ReflectionPad2d(self.padding)
        if(self.kernel_size==1):
            a = np.array([1.,])
        elif(self.kernel_size==2):
            a = np.array([1., 1.])
        elif(self.kernel_size==3):
            a = np.array([1., 2., 1.])
        elif(self.kernel_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.kernel_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.kernel_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.kernel_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        # self.register_buffer('filt', filt[None,None,:,:].repeat(self.channels,1,1,1))
        filt = filt[None,None,:,:].repeat(self.channels,1,1,1)
        self.weight = nn.Parameter(data=filt, requires_grad=True)
        ## Bayesian Frequency Matrix (BFM) 
        # alpha = 8
        # BFM = np.zeros((self.channels, 1, self.kernel_size, self.kernel_size))
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         point_1 = (i, j)
        #         point_2 = (self.kernel_size//2, self.kernel_size//2)
        #         dist = distance.euclidean(point_1, point_2)
        #         BFM[:, :, i, j] = alpha / (dist + alpha)
        # self.register_buffer('BFM', torch.Tensor(BFM))
        # self.BFM = nn.Parameter(data=torch.Tensor(BFM), requires_grad=False)
        # self.BFM = torch.from_numpy(BFM).type(torch.cuda.FloatTensor)

    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        # if self.training:
            # with torch.no_grad():
                # https://discuss.pytorch.org/t/set-nn-parameter-during-training/178163/3
                # self.weight.copy_(self.weight * self.BFM)
        # print(self.weight)
        x = self.PAD(x)
        x = F.conv2d(x, 
                     self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, self.kernel_size, self.kernel_size), 
                    #  self.weight.abs() / self.weight.abs().sum(dim=(-1, -2), keepdims=True), 
                    padding=0, groups=self.channels, stride=self.stride)
        return x
    
def test_DCNv2_AS():
    m = DCNv2_AS(in_channels=2,
                 out_channels=2,
                 kernel_size=5,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=2,
                 offset_size=3,
                 offset_kernel=3,
                 offset_dilation=5,
                 bias=True, radius=2)
    m.weight.data.fill_(1)
    print(m.weight)
    # kernel:
    # (x0, y0) (x1, y1) (x2, y2)
    # (x3, y3) (x4, y4) (x5, y5)
    # (x6, y6) (x7, y7) (x8, y8)
    #  [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
    x1 = torch.Tensor([
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        ])
    x2 = torch.Tensor([
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        [1, 2, 4, 7, 11],
        ]) * 2
    # x = x [None, None, ].repeat(1, 2, 1, 1)
    x = torch.stack([x1, x2], dim=0)[None, ]
    offset = None
    offset = torch.zeros(1, 18 * 2, 5, 5) # [b, group * [y, x, y, x, ...], h, w]
    # offset[0, 0, 1, 1] = 1
    # offset[0, 1, 1, 1] = 1
    # offset[0, 2, 1, 1] = 1
    # offset[0, 18 * 0 + 4 * 2 + 1, 1, 1] = 1
    offset[0, 18 * 1 + 4 * 2 + 1, 1, 1] = 2
    # print(offset)
    print(m(x, offset))
    pass

def test_cat():
    x = torch.zeros(1, 9, 3, 3)
    y = torch.ones(1, 9, 3, 3)
    # torch.fill_(1)
    # pass
    # xy = torch.cat([x, y], 1)
    xy = torch.stack([x, y], 2)
    print(xy.shape)
    xy = xy.reshape(1, 18, 3, 3) # reshape has direction, increase dim ->, reduce dim <-
    print(xy)

if __name__=='__main__':
    # test_2()
    # test_coordinate_generate()
    # test_torch_patchwise_dct()
    # test_SaliencySampler()
    # test_makeGaussian()
    # test_softpool()
    # test_makeGaussian()
    # test_InformationEntropy()
    # test_MultualInformation()
    # test_LocalCosSim()
    test_DCNv2_AS()
    # test_cat()

