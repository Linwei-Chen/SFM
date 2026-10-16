
import torch
import torch.nn as nn
from mmseg.ops import resize
# import torch_dct as dct
import torch.fft as fft
import torch.nn.functional as F
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, ModulatedDeformConv2dPack
from mmcv.ops.carafe import CARAFEPack, normal_init, xavier_init, carafe
from torch.utils.checkpoint import checkpoint
import numpy as np
import os as os

def torch_patchwise_dct(img, log=False):
    import torch_dct as dct
    # x = torch.randn(128, 128)
    if not isinstance(img, torch.Tensor):
        x = torch.Tensor(img)
    else:
        x = img
    assert x.dim() == 4
    b, c, h, w = x.shape
    # print(x.shape)
    # print(x.dim())
    # print(h - (h%8), w - (w%8))
    x = x[:, :, :h - (h%8), :w - (w%8)]
    x = F.unfold(x, kernel_size=(8, 8), stride=8, dilation=1)
    print('shape after 8x8 unfold', x.shape)
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
    def __init__(self, save_path='/home/ubuntu/code/ResolutionDet/mmseg_exp/utils/zigzag_img2_8x_dct.npy'):
        self.dct_list = []
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def save(self, x, print_avg=True, log=False):
        self.dct_list.append(x)
        avg = torch.stack(self.dct_list, dim=0).mean(dim=0)
        if log:
            avg = avg.log()
            avg[avg < 0] = 0
        avg = avg.cpu().numpy()
        if print_avg:print(self.save_path, len(self.dct_list), '\n', avg)
        np.save(self.save_path, avg)

    def get_avg(self):
        avg = torch.stack(self.dct_list, dim=0).mean(dim=0)
        return avg

class DCTSpatialFusion(nn.Module):
    def __init__(self, kernel_size=7, align_corners=False, use_freq_domain=False, upsample_mode='bilinear'):
        super().__init__()
        self.conv1 = nn.Conv2d(4 + 2, 1, kernel_size, padding=0, bias=False)
        self.pad0 = nn.ZeroPad2d(kernel_size // 2)

        # self.sigmoid = nn.Sigmoid()
        # self.coord_grid = None 
        self.coord_grids = {} 
        self.align_corners = align_corners
        self.padding_size = kernel_size // 2
        self.use_freq_domain = use_freq_domain
        self.sigmax = torch.nn.parameter.Parameter(torch.tensor([1.0]))
        self.sigmay = torch.nn.parameter.Parameter(torch.tensor([1.0]))
        self.upsample_mode = upsample_mode

    def gen_gaussian_mask(self, h, w, device):
        coord = self.gen_coord_grid(h, w, device)
        x, y = coord[0:1], coord[1:2]
        # x0 = 0,
        x0, y0 = 0, 0
        g = torch.exp(-4 * 0.6931471805599453 * ((x-x0)**2 / self.sigmax ** 2 + (y-y0)**2 / self.sigmay))
        return g

    def gen_coord_grid(self, h, w, device=None):
        """
        return: [2, h, w]
        """
        coord_grid = self.coord_grids.get(f'{h}x{w}', None)
        if coord_grid == None:
        # if self.coord_grid == None \
        #     or self.coord_grid.shape[-2] != (h + 2 * self.padding_size) \
        #         or self.coord_grid.shape[-1] != (w + 2 * self.padding_size):
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
            self.coord_grids[f'{h}x{w}'] = coord_grid
            # self.coord_grid = coord_grid
        else:
            pass
        # print(coord_grid.requires_grad)
        return coord_grid.detach()
        # return self.coord_grid

    def get_max_avg_cat(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = dct.dct_2d(avg_out)
        # max_out = dct.dct_2d(avg_out)
        x = torch.cat([avg_out, max_out], dim=1)
        return x

    def forward(self, x_hr, x_lr):
        if self.use_freq_domain:
            return self.frequency_forward(x_hr, x_lr)
        else:
            return self.spatial_forward(x_hr, x_lr)

    def spatial_forward(self, x_hr, x_lr):
        x_lr = resize(
            input=x_lr,
            size=x_hr.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        x_hr_avg_max_cat = self.get_max_avg_cat(x_hr)
        x_lr_avg_max_cat = self.get_max_avg_cat(x_lr)
        x_hr_dct = dct.dct_2d(x_hr_avg_max_cat)
        x_lr_dct = dct.dct_2d(x_lr_avg_max_cat)
        x_hr_dct = self.pad0(x_hr_dct)
        x_lr_dct = self.pad0(x_lr_dct)
        b, _, h, w = x_hr.shape
        coord = self.gen_coord_grid(h, w, device=x_hr.device)
        coord = coord[None, ].repeat(b, 1, 1, 1)
        att = torch.cat([coord, x_hr_dct, x_lr_dct], dim=1)
        att = self.conv1(att)
        att = dct.idct_2d(att)
        att = att.sigmoid()
        return att, x_hr * att, x_lr * (1.0 - att)

    def _spatial_forward(self, x_hr, x_lr):
        x_lr = resize(
            input=x_lr,
            size=x_hr.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x_hr_avg_max_cat = self.get_max_avg_cat(x_hr)
        x_lr_avg_max_cat = self.get_max_avg_cat(x_lr)
        # x_hr_dct = dct.dct_2d(x_hr_avg_max_cat)
        # x_lr_dct = dct.dct_2d(x_lr_avg_max_cat)
        x_hr_dct = self.pad0(x_hr_avg_max_cat)
        x_lr_dct = self.pad0(x_lr_avg_max_cat)
        b, _, h, w = x_hr.shape
        coord = self.gen_coord_grid(h, w, device=x_hr.device)
        coord = coord[None, ].repeat(b, 1, 1, 1)
        att = torch.cat([coord, x_hr_dct, x_lr_dct], dim=1)
        att = self.conv1(att)
        # att = dct.idct_2d(att)
        att = att.sigmoid()
        return att, x_hr * att, x_lr * (1.0 - att)
        # return att, x_hr * att[:, 0:1], x_lr * att[:, 1:2]

    def frequency_forward(self, x_hr, x_lr):
        x_lr = resize(
            input=x_lr,
            size=x_hr.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x_hr_dct = dct.dct_2d(x_hr)
        x_lr_dct = dct.dct_2d(x_lr)
        # x_hr_dct = fft.fft2(x_hr)
        # x_lr_dct = fft.fft2(x_lr)
        # x_lr_dct = fft.fftshift(fft.fft2(x_lr))
        # x_hr_dct = torch.rfft(x_hr, signal_ndim=2, normalized=False, onesided=False)
        # x_lr_dct = torch.rfft(x_lr, signal_ndim=2, normalized=False, onesided=False)
        # x_hr_dct_max_avg_cat = x_hr_dct.mean(dim=1, keepdim=True)
        # x_lr_dct_max_avg_cat = x_lr_dct.mean(dim=1, keepdim=True)
        # x_hr_dct_max_avg_cat = self.get_max_avg_cat(x_hr_dct)
        # x_lr_dct_max_avg_cat = self.get_max_avg_cat(x_lr_dct)
        # x_hr_dct_pad = self.pad0(x_hr_dct_max_avg_cat)
        # x_lr_dct_pad = self.pad0(x_lr_dct_max_avg_cat)
        # x_hr_dct_pad = self.pad0(x_hr_dct)
        # x_lr_dct_pad = self.pad0(x_lr_dct)
        b, _, h, w = x_hr.shape
        # coord = self.gen_coord_grid(h, w, device=x_hr.device)
        # coord = coord[None, ].repeat(b, 1, 1, 1)
        # att = torch.cat([coord, x_hr_dct_pad, x_lr_dct_pad], dim=1)
        # att = self.conv1(att)
        # att = dct.idct_2d(att)
        # att = att.sigmoid()
        # att = att.softmax(dim=1)
        # print(x_hr_dct.shape, x_lr_dct.shape, att.shape)
        att = self.gen_gaussian_mask(h, w, device=x_hr.device)
        att = att[None, ].repeat(b, 1, 1, 1)
        return att, dct.idct_2d(x_hr_dct * (1.0 - att)), dct.idct_2d(x_lr_dct * att)
        # return att, dct.idct_2d(x_hr_dct * att), dct.idct_2d(x_lr_dct * (1.0 - att))
        # return att, dct.idct_2d(x_hr_dct * att[:, 0:1]), dct.idct_2d(x_lr_dct *  att[:, 1:2])
        # return att, fft.ifft2(fft.ifftshift(x_hr_dct * att)).real, fft.ifft2(fft.ifftshift(x_lr_dct * (1.0 - att))).real

class DCTFrequencySoftSelection(DCTSpatialFusion):
    def __init__(self, kernel_size=7, align_corners=False, use_freq_domain=False):
        super().__init__(kernel_size, align_corners)
        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=0, bias=False)
        self.bias = nn.parameter.Parameter(torch.Tensor([1]))
        self.pad0 = nn.ZeroPad2d(kernel_size // 2)

        # self.sigmoid = nn.Sigmoid()
        self.coord_grid = None 
        self.align_corners = align_corners
        self.padding_size = kernel_size // 2
        self.use_freq_domain = use_freq_domain

    def forward(self, x_hr):
        if self.use_freq_domain:
            return self.frequency_forward(x_hr)
        else:
            return self.spatial_forward(x_hr)

    def frequency_forward(self, x_hr):
        b, _, h, w = x_hr.shape
        # x_hr_dct = unfold(x_hr, 8)
        # x_hr_dct = torch.fft.fftshift(torch.fft.fft2(x_hr))
        x_hr_dct = dct.dct_2d(x_hr)
        # x_hr_dct = dct.dct_2d(x_hr_dct)
        # x_hr_dct = fold(x_hr_dct, h, w)
        x_hr_avg_max_cat = self.get_max_avg_cat(x_hr_dct)
        x_hr_avg_max_cat = self.pad0(x_hr_avg_max_cat)
        coord = self.gen_coord_grid(h, w, device=x_hr.device).flip(dims=(-1, -2))
        coord = coord[None, ].repeat(b, 1, 1, 1)
        att = torch.cat([coord, x_hr_avg_max_cat], dim=1)
        att = self.conv1(att)
        # att = self.conv1(coord)
        # att = dct.idct_2d(att)
        att = att.sigmoid()
        att[:, :, :h//2, :w//2 ] = att[:, :, :h//2, :w//2 ].detach()
        att[:, :, :h//2, :w//2 ] = 1
        out = att * x_hr_dct
        out = dct.idct_2d(out)
        # print(self.bias)
        # print(att.mean())
        return att, out
        
    def spatial_forward(self, x_hr):
        b, _, h, w = x_hr.shape
        x_hr_avg_max_cat = self.get_max_avg_cat(x_hr)
        # x_hr_avg_max_cat = unfold(x_hr_avg_max_cat, 8)
        x_hr_dct = dct.dct_2d(x_hr_avg_max_cat)
        # x_hr_dct = x_hr_avg_max_cat
        # x_hr_dct = fold(x_hr_dct, h, w)
        x_hr_dct = self.pad0(x_hr_dct)
        # coord = self.gen_coord_grid(h, w, device=x_hr.device).flip(dims=(-1, -2))
        coord = self.gen_coord_grid(h, w, device=x_hr.device)
        coord = coord[None, ].repeat(b, 1, 1, 1)
        att = torch.cat([coord, x_hr_dct], dim=1)
        att = self.conv1(att)
        att[:, :, :h//2, :w//2 ] = 0
        att = dct.idct_2d(att)
        att = att.sigmoid()
        return att, x_hr * att
        return att, x_hr


def unfold(output, patch_size):
    b, c, h, w = output.shape
    # print(output.shape)
    output = torch.nn.functional.unfold(output, (patch_size , patch_size), stride=patch_size) # c, c * 8 * 8, h // 8, w // 8
    output = output.reshape(b, c, patch_size * patch_size, h // patch_size, w // patch_size)
    output = output.permute(0, 1, 3, 4, 2)
    output = output.reshape(b, -1, patch_size, patch_size)

    # output = output.reshape(b, c, h // patch_size, w // patch_size, patch_size * patch_size)
    # output = output.permute(0, 1, 4, 2, 3)
    # output = output.reshape(b, c * patch_size * patch_size, h // patch_size * w // patch_size)
    # output = torch.nn.functional.fold(output, (h, w), (patch_size, patch_size), stride=patch_size)
    # print(output.shape)
    
    return output # (b, c * h // patch_size * 8 * w // patch_size, patch_size, patch_size,)

def fold(output, h, w):
    b, c, patch_size, patch_size = output.shape
    c = c * patch_size * patch_size // h // w
    # c, h, w = (c // patch_size ** 2), h * patch_size, w * patch_size
    output = output.reshape(b, c, h // patch_size, w // patch_size, patch_size * patch_size)
    output = output.permute(0, 1, 4, 2, 3)
    output = output.reshape(b, c * patch_size * patch_size, h // patch_size * w // patch_size)
    output = torch.nn.functional.fold(output, (h, w), (patch_size, patch_size), stride=patch_size)
    # print(output.shape)
    return output

class SpatialGateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_random()

    def forward(self, hr_feat, lr_feat):
        return torch.sigmoid(F.interpolate(self.conv(lr_feat), size=hr_feat.shape[2:], mode='nearest'))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


############ Dysample
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
    
class DyUpSample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1):
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

        self.offset = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

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
    
    def forward(self, offset_feat, x):
        offset = self.get_offset(offset_feat)
        return self.sample(x, offset)
    
    def get_offset_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return offset

    def get_offset_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return offset

    def get_offset(self, x):
        if self.style == 'pl':
            return self.get_offset_pl(x)
        return self.get_offset_lp(x)
    
class FusionDyUpSample(DyUpSample):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1):
        super().__init__(in_channels=in_channels, scale=scale, style=style, groups=groups, dyscope=dyscope, kernel_size=kernel_size)
        assert scale==2
        assert style=='lp'
        out_channels = 2 * groups
        self.hr_offset = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.hr_scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

    def forward(self, hr_x, lr_x, feat2sample):
        offset = self.get_offset(hr_x, lr_x)
        return super().sample(feat2sample, offset)
    
    def get_offset_lp(self, hr_x, lr_x):
        if hasattr(self, 'scope'):
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.scope(lr_x) + F.pixel_unshuffle(self.hr_scope(hr_x), self.scale)).sigmoid() * 0.5 + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)

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

class ConvSaliencySampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, coord_mode='yx', out_mode='abs_offset', 
                 gaussian_radius=2, padding_size=2, padding_mode='replicate'):
        super().__init__()
        self.align_corners = align_corners
        self.dilation = dilation
        self.gaussian_radius = gaussian_radius
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.saliency_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        # self.saliency_conv_hr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        # self.saliency_conv_lr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.eps = 1e-16
        self.coord_mode = coord_mode
        self.out_mode = out_mode # abs_offset for deformconv, rel_grid for grid_sample
        self.init_gaussian_filter()
        # for fusion
        assert self.coord_mode == 'xy'
        assert self.out_mode == 'rel_grid'

    def init_gaussian_filter(self):
        gaussian_weights = torch.Tensor(makeGaussian(2 * self.padding_size + 1, fwhm = self.gaussian_radius))
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
        saliency = self.saliency_conv(feature)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        offsets = self.create_conv_offset(saliency, h, w) # b, h, w, 2
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
        x: saliency in [b, g, h, w]
        """
        s = F.pad(s, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size
        P = self.gen_coord_grid(h, w, device=s.device).reshape(1, 2, padded_h, padded_w)
        P = P.expand(s.size(0), -1, padded_h, padded_w) # b, 2, ph, pw
        P = P.repeat(1, s.size(1), 1, 1) # b, 2 * g, ph, pw

        p_filter = self.Filter(s.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter = p_filter.view(s.size(0), -1, h, w) # b, g, ph, pw
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
        x_filter = self.Filter(x_filter).view(s.size(0), -1, h, w) # b, g, ph, pw
        y_filter = torch.mul(P[:,1::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        y_filter = self.Filter(y_filter).view(s.size(0), -1, h, w) # b, g, ph, pw

        #值的范围是[0,1]
        # print(x.min(), x.max(), x.mean())
        # x_filter = x_filter / (p_filter + 1e-16) #u(x,y)
        # y_filter = y_filter / (p_filter + 1e-16) #v(x,y)
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter + self.eps) #v(x,y)
        if 'abs_offset' ==  self.out_mode:
            x_offsets = (x_filter - P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * w
            y_offsets = (y_filter - P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * h
        elif 'rel_grid' ==  self.out_mode:
            x_offsets = x_filter * 2.0 - 1.0
            y_offsets = y_filter * 2.0 - 1.0
        else:
            raise NotImplementedError
        if self.coord_mode == 'xy': # dcnv3, grid_sample
            grid = torch.stack((x_offsets, y_offsets), 2) #[batch_size, position, 2, h, w]
        elif self.coord_mode == 'yx': # dcnv2
            grid = torch.stack((y_offsets, x_offsets), 2) #[batch_size, position, 2, h, w]
        else:
            raise NotImplementedError
        grid = grid.reshape(s.size(0), -1, h, w) #[batch_size, position, 2, h, w]
        return grid
    
    def sample_with_grid(self, feature, grid):
        feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled
    
class FusionSaliencySampler(ConvSaliencySampler):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, 
                 gaussian_radius=2, padding_size=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, align_corners=align_corners, coord_mode='xy', out_mode='rel_grid', gaussian_radius=gaussian_radius, padding_size=padding_size)
        del self.saliency_conv
        self.saliency_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.saliency_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)

        self.conv_att_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.conv_att_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        normal_init(self.saliency_conv_1, std=0.001)
        normal_init(self.saliency_conv_2, std=0.001)
        constant_init(self.conv_att_1, val=0.)
        constant_init(self.conv_att_2, val=0.)

    def compressed_feat_forward(self, hr_x, lr_x, mask_lr, up_kernel, up_lr_x):
        """
        hr_x, lr_x, compressed feature
        """
        saliency = carafe(self.saliency_conv_1(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_1(hr_x)
        saliency2 = carafe(self.saliency_conv_2(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_2(hr_x)
        # att1 = carafe(self.conv_att_1(lr_x), mask_lr, up_kernel, 1, 2) + self.conv_att_1(hr_x)
        # att2 = carafe(self.conv_att_2(lr_x), mask_lr, up_kernel, 1, 2) + self.conv_att_2(hr_x)
        # att1 = att1.sigmoid()
        # att2 = att2.sigmoid()
        # saliency = F.pixel_shuffle(self.saliency_conv_lr(lr_x), upscale_factor=2) + self.saliency_conv_hr(hr_x)
        # saliency2 = F.pixel_shuffle(self.saliency_conv_lr_2(lr_x), upscale_factor=2) + self.saliency_conv_hr_2(hr_x)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        saliency2 = saliency2.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        # saliency2 = saliency
        grid = self.create_conv_offset(saliency, saliency2, 1, 1, h, w)
        grid = grid.reshape(-1, 2, h, w)
        group = c
        feature_sampled = F.grid_sample(up_lr_x.reshape(b * group, -1, h, w), grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled.reshape(b, -1, h, w)

    def create_conv_offset(self, s, s2, att1, att2, h, w):
        """
        x: saliency in [b, g, h, w]
        """
        s = F.pad(s, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        s2 = F.pad(s2, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size

        P = self.gen_coord_grid(h, w, device=s.device).reshape(1, 2, padded_h, padded_w)
        P = P.expand(s.size(0), -1, padded_h, padded_w) # b, 2, ph, pw
        P = P.repeat(1, s.size(1), 1, 1) # b, 2 * g, ph, pw

        p_filter = self.Filter(s.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter = p_filter.view(s.size(0), -1, h, w) # b, g, ph, pw
        
        p_filter2 = self.Filter(s2.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter2 = p_filter2.view(s2.size(0), -1, h, w) # b, g, ph, pw
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
        x_filter = self.Filter(x_filter).view(s.size(0), -1, h, w) # b, g, ph, pw
        y_filter = torch.mul(P[:,1::2,:,:].view(s2.size(0), -1, padded_h, padded_w), s2).view(-1, 1, padded_h, padded_w)
        y_filter = self.Filter(y_filter).view(s2.size(0), -1, h, w) # b, g, ph, pw

        #值的范围是[0,1]
        # print(x.min(), x.max(), x.mean())
        # x_filter = x_filter / (p_filter + 1e-16) #u(x,y)
        # y_filter = y_filter / (p_filter + 1e-16) #v(x,y)
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter2 + self.eps) #v(x,y)
        if 'abs_offset' ==  self.out_mode:
            x_offsets = (x_filter - P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * w
            y_offsets = (y_filter - P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * h
        elif 'rel_grid' ==  self.out_mode:
            grid_x = P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            grid_y = P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            x_offsets = (x_filter - grid_x) * att1 + grid_x
            y_offsets = (y_filter - grid_y) * att2 + grid_y
            x_offsets = x_offsets * 2.0 - 1.0
            y_offsets = y_offsets * 2.0 - 1.0
            # x_offsets = x_filter * 2.0 - 1.0
            # y_offsets = y_filter * 2.0 - 1.0
        else:
            raise NotImplementedError
        if self.coord_mode == 'xy': # dcnv3, grid_sample
            grid = torch.stack((x_offsets, y_offsets), 2) #[batch_size, position, 2, h, w]
        elif self.coord_mode == 'yx': # dcnv2
            grid = torch.stack((y_offsets, x_offsets), 2) #[batch_size, position, 2, h, w]
        else:
            raise NotImplementedError
        grid = grid.reshape(s.size(0), -1, h, w) #[batch_size, position, 2, h, w]
        return grid

    def align_forward(self, hr_x, lr_x, mask_lr, up_kernel):
        saliency = carafe(self.saliency_conv_lr(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_hr(hr_x)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        grid = self.create_conv_offset(saliency, h, w)
        feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled


def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)

    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)

    return similarity

class LocalSimGuidedSampler(DyUpSample):
    # class FusionDyUpSample(DyUpSample):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1, local_window=3, sim_type='cos', norm=True):
        super().__init__(in_channels=in_channels, scale=scale, style=style, groups=groups, dyscope=dyscope, kernel_size=kernel_size)
        assert scale==2
        assert style=='lp'

        self.local_window = local_window
        self.sim_type = sim_type

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        # self.offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            # self.scope = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            # self.scope = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        out_channels = 2 * groups
        # self.hr_offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.hr_offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        
        if dyscope:
            # self.hr_scope = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            # self.hr_scope = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.hr_scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)
        hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
        lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
        # hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
        # lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        # offset = self.get_offset(hr_x, lr_x)
        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return super().sample(feat2sample, offset)
    
    # def get_offset_lp(self, hr_x, lr_x):
    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'scope'):
            # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.scope(lr_x) + F.pixel_unshuffle(self.hr_scope(hr_x), self.scale)).sigmoid() + self.init_pos
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.scope(lr_x) + F.pixel_unshuffle(self.hr_scope(hr_x), self.scale)).sigmoid() + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)
    
class SaliencyDySampler(ConvSaliencySampler):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, 
                 gaussian_radius=2, padding_size=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, align_corners=align_corners, coord_mode='xy', out_mode='rel_grid', gaussian_radius=gaussian_radius, padding_size=padding_size)
        del self.saliency_conv
        # self.saliency_conv_hr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.saliency_conv_lr = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
    
    def forward(self, x):
        saliency = self.saliency_conv_lr(x)
        saliency = F.pixel_shuffle(saliency, upscale_factor=2)
        # print(saliency.shape)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        grid = self.create_conv_offset(saliency, h, w)
        grid = grid.reshape(-1, 2, h, w)
        group = c
        feature_sampled = F.grid_sample(x.reshape(b * group, -1, x.size(-2), x.size(-1)), grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled.reshape(b, -1, h, w)
    
##########
class GobalContextLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int,  c_n: int, c_m: int = 0, reconstruct = False):
        """

        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super().__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        # self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, compressed_x: torch.Tensor, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, h, w = x.size()
        # assert c == self.in_channels, 'input channel not equal!'
        # A = self.convA(compressed_x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(compressed_x)  # (B, c_n, h, w)
        V = self.convV(compressed_x)  # (B, c_n, h, w)
        tmpA = x.view(batch_size, c, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        # tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        tmpZ = tmpZ.view(batch_size, c, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

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
        g = g.reshape(b, c, 1, 1)
        # return x * g
        return g
    
class EdgeConvAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=1, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.c_att = SRMLayer(in_planes)
        # self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        # self.func_channel = self.get_channel_attention

        # if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
        #     self.func_filter = self.skip
        # else:
        #     self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
        #     self.func_filter = self.get_filter_attention

        # if kernel_size == 1:  # point-wise convolution
        #     self.func_spatial = self.skip
        # else:
        #     self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        #     self.func_spatial = self.get_spatial_attention

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

    # def get_channel_attention(self, x):
    #     channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
    #     return channel_attention

    # def get_filter_attention(self, x):
    #     filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
    #     return filter_attention

    # def get_spatial_attention(self, x):
    #     spatial_attention = self.spatial_fc(x).reshape(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
    #     # spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
    #     spatial_attention = torch.sigmoid(spatial_attention)
    #     return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).reshape(x.size(0), -1, 1, 1, 1, 1)
        # kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        kernel_attention = F.softmax(kernel_attention, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x) * self.c_att(x)
        # x = self.avgpool(x)
        # x = self.fc(x)
        # x = self.bn(x)
        x = self.relu(x)
        # return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
        return 1, 1, 1, self.func_kernel(x)

class DyEdgeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 reduction=1, kernel_num=5):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = EdgeConvAttention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=5)
        # self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size, kernel_size) * 0.001, requires_grad=True) # 1, 1 for batch and kernel num
        self.bias = nn.Parameter(torch.zeros(out_planes), requires_grad=True)
        self.weight_lr_diag = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size) * 0.001, requires_grad=True)
        self.weight_rl_diag = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size) * 0.001, requires_grad=True)
        self.weight_h = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, 1, kernel_size) * 0.001, requires_grad=True)
        self.weight_v = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size, 1) * 0.001, requires_grad=True)
        # normal_init(self.content_encoder2, std=0.001)
        # self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        # print('kernel_attention', kernel_attention.shape)
        # print('spatial_attention', spatial_attention.shape)
        batch_size, in_planes, height, width = x.size()
        # x = x * channel_attention
        x = x.reshape(1, -1, height, width) # 1, b*c , h, w 
        # spatial_attention (b, 1, 1, 1, self.kernel_size, self.kernel_size)
        # kernel_attention (b, -1, 1, 1, 1, 1)
        # aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        weight = kernel_attention[:, 0:1] * self.weight
        weight_lr_diag = kernel_attention[:, 1:2, ..., 0] * self.weight_lr_diag
        weight_rl_diag = kernel_attention[:, 2:3, ..., 0] * self.weight_rl_diag
        weight_h = kernel_attention[:, 3:4] * self.weight_h
        weight_v = kernel_attention[:, 4:5] * self.weight_v

        weight += torch.diag_embed(weight_lr_diag)
        # print(torch.diag_embed(weight_lr_diag))
        weight += torch.flip(torch.diag_embed(weight_rl_diag), dims=[-1])
        # print(torch.flip(torch.diag_embed(weight_rl_diag), dims=[-1]))
        weight[..., self.kernel_size//2:self.kernel_size//2+1, :] += weight_h
        weight[..., self.kernel_size//2:self.kernel_size//2+1] += weight_v
        # print('weight', weight.shape)
        
        # aggregate_weight = spatial_attention * weight
        aggregate_weight = weight
        # print('aggregate_weight', aggregate_weight.shape)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        # print(aggregate_weight.shape) # batch * out_c, in_c, kh, kw
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1)) + self.bias.view(1, -1, 1, 1)
        # print(self.bias)
        # output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        # print('training', self.training)
        return self._forward_impl(x)
    
class FreqFusion2encoder(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor=1,
                lowpass_kernel=5,
                highpass_kernel=3,
                lowpass_pad=0,
                highpass_pad=0,
                padding_mode='replicate',
                hamming_window=True,
                up_group=1,
                temp_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                feature_align=False,
                feature_align_group=4,
                comp_feat_upsample=True,
                # use_encoder2=False,
                use_spatial_suppression=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                hf_att=False,
                semi_conv=False,
                use_spatial_gate=False,
                use_global_context=False,
                use_channel_att=False,
                use_dyedgeconv=False,
                **kwargs):
        super().__init__()
        # self.channels = channels
        self.scale_factor = scale_factor
        # self.up_kernel = up_kernel

        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.lowpass_pad = lowpass_pad
        self.highpass_pad = highpass_pad
        self.padding_mode = padding_mode
        self.hamming_window = hamming_window
        self.use_channel_att = use_channel_att
        self.use_dyedgeconv = use_dyedgeconv
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
            # self.hamming_lowpass = 1
            # self.hamming_highpass = 1
        self.up_group = up_group
        self.temp_group = temp_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)
        if self.use_channel_att:
            self.hr_channel_att = SRMLayer(hr_channels)
            self.lr_channel_att = SRMLayer(lr_channels)
        if self.use_dyedgeconv:
            self.content_encoder = DyEdgeConv2d(self.compressed_channels, 
                                                lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0), 
                                                kernel_size=self.encoder_kernel, stride=1, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2), dilation=self.encoder_dilation, groups=1, reduction=1, kernel_num=5)
        else:
            self.content_encoder = nn.Conv2d(
                self.compressed_channels,
                lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0),
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        # self.lr_temp_encoder = None
        # if self.temp_group > 1:
        #     self.lr_temp_encoder = nn.Conv2d(
        #                 self.compressed_channels,
        #                 self.temp_group,
        #                 self.encoder_kernel,
        #                 padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
        #                 dilation=self.encoder_dilation,
        #                 groups=1)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.use_spatial_gate = use_spatial_gate
        self.semi_conv = semi_conv
        self.feature_align = feature_align
        self.comp_feat_upsample = comp_feat_upsample
        if self.feature_align:
            # self.lr_dysampler = AlignUpSample(in_channels=compressed_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=encoder_kernel)
            # self.hr_dysampler = AlignUpSample(in_channels=compressed_channels, scale=1, style='lp', groups=4, dyscope=True, kernel_size=encoder_kernel)
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp', groups=feature_align_group, dyscope=True, kernel_size=encoder_kernel)
            # self.dysampler = FusionSaliencySampler(in_channels=compressed_channels, out_channels=feature_align_group, kernel_size=3, dilation=1, align_corners=False, gaussian_radius=2, padding_size=2)
        if self.use_high_pass:
            if self.use_dyedgeconv:
                self.content_encoder2 = DyEdgeConv2d(self.compressed_channels, 
                                                     highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0), 
                                                     kernel_size=self.encoder_kernel, stride=1, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2), dilation=self.encoder_dilation, groups=1, reduction=1, kernel_num=5)
            else:
                self.content_encoder2 = nn.Conv2d(
                    self.compressed_channels,
                    highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0),
                    self.encoder_kernel,
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
            if hf_att:
                self.hf_att_hr_conv = nn.Conv2d(
                    self.compressed_channels,
                    1, 
                    self.encoder_kernel, 
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                self.hf_att_lr_conv = nn.Conv2d(
                    self.compressed_channels,
                    1, 
                    self.encoder_kernel, 
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                normal_init(self.hf_att_hr_conv, bias=2)
                normal_init(self.hf_att_lr_conv, bias=-2)
            # self.hr_temp_encoder = None
            # if self.temp_group > 1:
            #     self.hr_temp_encoder = nn.Conv2d(
            #             self.compressed_channels,
            #             self.temp_group,
            #             self.encoder_kernel,
            #             padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            #             dilation=self.encoder_dilation,
            #             groups=1)
        if use_global_context:
            self.global_context_layer = GobalContextLayer(in_channels=compressed_channels, c_n=compressed_channels)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        if use_spatial_gate:
            self.sp_gate = SpatialGateGenerator(in_channels=lr_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, padding=0, hamming=1, scale_factor=None):
        if self.temp_group > 1: 
            temp = mask[:, -self.temp_group:]
            mask = mask[:, :-self.temp_group]
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        # mask_channel = int(mask_c / float(self.up_kernel**2))
        mask_channel = int(mask_c / float(kernel**2))
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()
        if self.temp_group > 1: 
            mask = mask.view(n, mask_channel, 1, -1, h, w)
            mask = mask.repeat(1, 1, self.temp_group, 1, 1, 1) * temp.reshape(n, 1, self.temp_group, 1, h, w)
            mask = F.softmax(mask, dim=-3, dtype=mask.dtype)
            mask = mask.view(n, -1, h, w).contiguous()
        else:
            if self.hamming_window:
                # mask = mask.view(n, mask_channel, kernel, kernel, h, w)
                # mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
                # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
                # mask = mask * hamming
                # # print(hamming)
                # # print(mask.shape)
                # mask = mask.view(n, mask_channel, h, w, -1).softmax(-1)
                # mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()

                mask = mask.view(n, mask_channel, -1, h, w)
                mask = F.softmax(mask, dim=2, dtype=mask.dtype)
                mask = mask.view(n, mask_channel, kernel, kernel, h, w)
                mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
                mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
                mask = mask * hamming
                mask /= mask.sum(dim=(-1, -2), keepdims=True)
                # print(hamming)
                # print(mask.shape)
                mask = mask.view(n, mask_channel, h, w, -1)
                mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
            else:
                mask = mask.view(n, mask_channel, -1, h, w)
                mask = F.softmax(mask, dim=2, dtype=mask.dtype)
                mask = mask.view(n, mask_c, h, w).contiguous()
        return mask

    # def feature_reassemble(self, x, mask):
    #     x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
    #     return x

    # def carafe_forward(self, x):
    #     compressed_x = self.channel_compressor(x)
    #     mask = self.content_encoder(compressed_x)
    #     mask = self.kernel_normalizer(mask)
    #     x = self.feature_reassemble(x, mask)
    #     return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        if self.use_channel_att:
            hr_feat *= self.hr_channel_att(hr_feat)
            lr_feat *= self.lr_channel_att(lr_feat)
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        if hasattr(self, 'global_context_layer'): # 'dyupsample'
            global_context_lr_feat = self.global_context_layer(compressed_lr_feat, lr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                # mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                # mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                # mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                # mask_lr_lr_feat = F.interpolate(
                #     carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                # mask_lr = mask_lr_hr_feat + mask_lr_lr_feat
                # if self.use_high_pass:
                #     mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                #     mask_hr_lr_feat = F.interpolate(
                #         carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                #     mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                #     mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, padding=self.highpass_pad, hamming=self.hamming_highpass)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel + 2 * self.highpass_pad, self.temp_group, 1)
                    
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else: raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass: 
                mask_hr = self.content_encoder2(compressed_x)
        
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
        if self.semi_conv:
            # if self.feature_align: # 'dyupsample'
            # lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
            if hasattr(self, 'hf_att_hr_conv') and hasattr(self, 'hf_att_lr_conv'): 
                lr_feat_lf = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
                lr_feat = F.interpolate(lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest')
                lr_feat_hf = lr_feat - lr_feat_lf
                lr_feat_hf_att = self.hf_att_lr_conv(compressed_hr_feat) + self.hf_att_lr_conv(F.interpolate(
                        carafe(compressed_lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest'))
                lr_feat_hf_att = torch.tanh(lr_feat_hf_att)
                lr_feat_hf = lr_feat_hf * lr_feat_hf_att
                lr_feat = lr_feat_hf + lr_feat
            else:
                lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 1)
            if self.feature_align:
                lr_feat = self.dysampler.sample(lr_feat, offset, scale=2)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, padding=self.highpass_pad, hamming=self.hamming_highpass)
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel + 2 * self.highpass_pad, self.temp_group, 1)
                if hasattr(self, 'hf_att_hr_conv') and hasattr(self, 'hf_att_lr_conv'): 
                    hr_feat_hf_att = self.hf_att_hr_conv(compressed_hr_feat) + self.hf_att_hr_conv(F.interpolate(
                            carafe(compressed_lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest'))
                    hr_feat_hf_att = torch.tanh(hr_feat_hf_att)
                    hr_feat_hf = hr_feat_hf * hr_feat_hf_att
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf
        # print(mask_lr.shape)
        # print(mask_hr.shape)
        if self.feature_align: # 'dyupsample'
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, 
                                     lr_x=compressed_lr_feat, feat2sample=lr_feat)
                
        if self.use_spatial_gate: 
            sp_gate = self.sp_gate(hr_feat, lr_feat)
            hr_feat *= sp_gate
            lr_feat *= (1.0 - sp_gate)
        # print(mask_lr.std(dim=[1, 2, 3]).mean())
        return  mask_lr, hr_feat, lr_feat

class FreqFusion(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor,
                up_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                semi_conv=False,
                **kwargs):
        super().__init__()
        # self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)

        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv

        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_encoder2:
            normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def carafe_forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        x = self.feature_reassemble(x, mask)
        return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # compressed_lr_feat = resize(
        #     input=compressed_lr_feat,
        #     size=hr_feat.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        # x = torch.cat([hr_feat, lr_feat], dim=1)
        # compressed_x = self.channel_compressor(x)
        if self.semi_conv:
            # mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), scale_factor=2, mode='nearest')
            mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[2:], mode='nearest')
        else:
            # compressed_x = F.interpolate(compressed_lr_feat, scale_factor=2, mode='nearest') + compressed_hr_feat
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[2:], mode='nearest') + compressed_hr_feat
            mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        # if self.use_spatial_suppression:
            # hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask) + hr_feat
            else:
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask)
        if self.use_low_pass:
            if self.use_encoder2:
                # mask2 = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), scale_factor=2)
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            if self.semi_conv:
                lr_feat = carafe(lr_feat, mask, self.up_kernel, self.up_group, 2)
            else:
                lr_feat = resize(
                    input=lr_feat,
                    size=hr_feat.shape[2:],
                    mode=self.upsample_mode,
                    align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
                lr_feat = self.feature_reassemble(lr_feat, mask2)
        return mask, hr_feat, lr_feat

class FixedFreqFusion(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor,
                up_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                **kwargs):
        super().__init__()
        self.low_pass = LLPFConv3(hr_channels, stride=1, padding=1)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        lr_feat = resize(
            input=lr_feat,
            size=hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            # lr_feat = carafe(lr_feat, mask, self.up_kernel, self.up_group, 2)
        hr_feat_hf = hr_feat - self.low_pass(hr_feat)
        lr_feat = self.low_pass(lr_feat)
        if self.hr_residual:
            hr_feat = hr_feat + hr_feat_hf
        return hr_feat, hr_feat, lr_feat
        
class DyHPFusion2(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                apply_hr_res_to_low=False,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.apply_hr_res_to_low = apply_hr_res_to_low
        if apply_hr_res_to_low:
            assert hr_residual == True
            assert use_high_pass == True
            assert use_low_pass == True

        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        self.init_weights()

    # def forward(self, x):
        # compressed_x = self.channel_compressor(x)
        # mask = self.content_encoder(compressed_x)
        # mask = self.kernel_normalizer(mask)
        # x = self.feature_reassemble(x, mask)
        # return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        lr_feat = resize(
            input=lr_feat,
            size=hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        x = torch.cat([hr_feat, lr_feat], dim=1)
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        if self.use_spatial_suppression:
            hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            hr_hi = hr_feat - self.feature_reassemble(hr_feat, mask)
            if self.hr_residual and not self.apply_hr_res_to_low:
                # print('using hr_residual')
                hr_feat = hr_hi + hr_feat
            else:
                hr_feat = hr_hi
        if self.use_low_pass:
            if self.use_encoder2:
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            lr_feat = self.feature_reassemble(lr_feat, mask2)
        if self.apply_hr_res_to_low: 
            lr_feat += hr_hi
        return mask, hr_feat, lr_feat

class FreqFusionFastEval(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels=None,
                lr_channels=None,
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                apply_hr_res_to_low=False,
                semi_conv=False,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        if hr_channels is None: hr_channels = self.channels // 2
        if lr_channels is None: lr_channels = self.channels // 2
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels

        self.apply_hr_res_to_low = apply_hr_res_to_low
        if apply_hr_res_to_low:
            assert hr_residual == True
            assert use_high_pass == True
            assert use_low_pass == True

        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        self.init_weights()

    # def forward(self, x):
        # compressed_x = self.channel_compressor(x)
        # mask = self.content_encoder(compressed_x)
        # mask = self.kernel_normalizer(mask)
        # x = self.feature_reassemble(x, mask)
        # return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        if not self.training:
            return self._forward_eval(hr_feat, lr_feat)
        lr_feat = resize(
            input=lr_feat,
            size=hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        x = torch.cat([hr_feat, lr_feat], dim=1)
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        if self.use_spatial_suppression:
            hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            hr_hi = hr_feat - self.feature_reassemble(hr_feat, mask)
            if self.hr_residual and not self.apply_hr_res_to_low:
                # print('using hr_residual')
                hr_feat = hr_hi + hr_feat
            else:
                hr_feat = hr_hi
        if self.use_low_pass:
            if self.use_encoder2:
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            lr_feat = self.feature_reassemble(lr_feat, mask2)
        if self.apply_hr_res_to_low: 
            lr_feat += hr_hi
        return mask, hr_feat, lr_feat

    def _forward_eval(self, hr_feat, lr_feat):
        # print(self.channel_compressor.weight)
        # compressed_hr_feat = F.conv2d(hr_feat, weight=self.channel_compressor.weight[:, self.hr_channels:])
        # compressed_lr_feat = F.conv2d(lr_feat, weight=self.channel_compressor.weight[:, :self.hr_channels, :])
        compressed_hr_feat = F.conv2d(hr_feat, weight=self.channel_compressor.weight[:, :self.hr_channels])
        compressed_lr_feat = F.conv2d(lr_feat, weight=self.channel_compressor.weight[:, self.hr_channels:, :])
        # compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # compressed_lr_feat = resize(
        #     input=compressed_lr_feat,
        #     size=hr_feat.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        # x = torch.cat([hr_feat, lr_feat], dim=1)
        # compressed_x = self.channel_compressor(x)
        if self.semi_conv:
            # mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), scale_factor=2, mode='nearest')
            mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[2:], mode='nearest')
        else:
            # compressed_x = F.interpolate(compressed_lr_feat, scale_factor=2, mode='nearest') + compressed_hr_feat
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[2:], mode='nearest') + compressed_hr_feat
            mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        # _lr_feat = resize(
        #     input=lr_feat,
        #     size=hr_feat.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # # if self.use_spatial_suppression:
        #     # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        # x = torch.cat([hr_feat, _lr_feat], dim=1)
        # _compressed_x = self.channel_compressor(x)
        # mask2 = self.content_encoder(_compressed_x)
        # mask2 = self.kernel_normalizer(mask2)
        # print((mask2 - mask).abs().mean())

        # if self.use_spatial_suppression:
            # hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask) + hr_feat
            else:
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask)
        if self.use_low_pass:
            if self.use_encoder2:
                # mask2 = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), scale_factor=2)
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            if self.semi_conv:
                lr_feat = carafe(lr_feat, mask, self.up_kernel, self.up_group, 2)
            else:
                lr_feat = resize(
                    input=lr_feat,
                    size=hr_feat.shape[2:],
                    mode=self.upsample_mode,
                    align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
                lr_feat = self.feature_reassemble(lr_feat, mask2)
        return mask, hr_feat, lr_feat

class _DyUpsample(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                hr_residual=False,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_spatial_residual = use_spatial_residual
        # self._do_nothing = nn.Conv2d(19, 1, 1)
        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.ss = SpatialSuppression(kernel_size=7, align_corners=align_corners, upsample_mode=upsample_mode)
        self.init_weights()

    def forward(self, feat, seg, img):
        img = resize(
            input=img,
            size=(feat.size(-2) * self.scale_factor, feat.size(-1) * self.scale_factor),
            mode='bilinear',
            align_corners=self.align_corners)
        # _ = self._do_nothing(seg)
        lr_img = resize(
            input=img,
            size=feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        lr_img = resize(
            input=lr_img,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        img_hf = img - lr_img
        # shape:  B x (C x ksize x ksize) x H // stride x W //stride
        b, _, h, w = feat.shape
        img_hf = F.unfold(img_hf, kernel_size=(self.scale_factor, self.scale_factor), stride=self.scale_factor, dilation=1).reshape(b, -1, h, w)
        compressed_x = self.channel_compressor(torch.cat([img_hf, feat], dim=1))
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        seg = self.feature_reassemble(seg, mask)
        # print(seg.shape)
        return seg

class CARAFEUpsample(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                # align_corners=False,
                # upsample_mode='nearest',
                # use_encoder2=False,
                # use_spatial_suppression=False,
                # use_spatial_residual=False,
                # hr_residual=False,
                **kwargs):
        super().__init__(**kwargs)
        # self.align_corners = align_corners
        # self.upsample_mode = upsample_mode
        # self.use_spatial_suppression = use_spatial_suppression
        # self.hr_residual = hr_residual
        # self.use_spatial_residual = use_spatial_residual
        # # self._do_nothing = nn.Conv2d(19, 1, 1)
        # self.use_encoder2 = use_encoder2
        # del self.channel_compressor
        # assert upsample_mode == 'nearest'
        # if self.use_encoder2:
        #     self.content_encoder2 = nn.Conv2d(
        #         self.compressed_channels,
        #         self.up_kernel * self.up_kernel * self.up_group *
        #         self.scale_factor * self.scale_factor,
        #         self.encoder_kernel,
        #         padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
        #         dilation=self.encoder_dilation,
        #         groups=1)
        # if self.use_spatial_suppression:
        #     self.ss = SpatialSuppression(kernel_size=7, align_corners=align_corners, upsample_mode=upsample_mode)
        # self.init_weights()

    def forward(self, feat, seg, img=None):
        # _ = self._do_nothing(seg)
        # feat_img = torch.cat([feat, img_hf], dim=1)
        compressed_x = self.channel_compressor(feat)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        seg = self.feature_reassemble(seg, mask)
        # feat = self.feature_reassemble(feat, mask)
        # print(seg.shape)
        return seg
        
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

class LLPFConv3(nn.Module):
    """
    Learnable Low Pass Filter
    """
    def __init__(self, channels=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.],
            # [1/8., 1/8., 1/8.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1).contiguous()
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        # identity_kernel = [
        #     [0., 0., 0.],
        #     [0., 1., 0.],
        #     [0., 0., 0.],
        # ]
        # identity_kernel = torch.FloatTensor(identity_kernel).unsqueeze(0).unsqueeze(0)
        # self.identity_kernel = nn.Parameter(data=identity_kernel, requires_grad=False)
 
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3), 
                    padding=self.padding, groups=self.channels, stride=self.stride)
        return x

class CascadeImageGuideUpsample(nn.Module):
    def __init__(self, n=2, **kargs):
        super().__init__()
        self.n = n
        self.align_corners = kargs.get('align_corners', False)
        self.up = nn.ModuleList()
        for _ in range(n):
            self.up.append(ImageGuideUpsample(**kargs))

    def forward(self, feat, seg, img):
        for i in range(self.n):
            resized_img = resize(
                input=img,
                size=(2 * feat.size(-2), 2 * feat.size(-1)),
                mode='bilinear',
                align_corners=self.align_corners)
            # print(resized_img.shape)
            seg, feat = self.up[i](feat, seg, resized_img)
            # print(seg.shape, feat.shape)
        return seg

class ImageGuideUpsample(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                hr_residual=False,
                use_deformable=False,
                use_leanable_hpf=False,
                use_extra_conv3=False,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_spatial_residual = use_spatial_residual
        # self._do_nothing = nn.Conv2d(19, 1, 1)
        self.use_deformable = use_deformable
        self.use_leanable_hpf = use_leanable_hpf
        self.use_extra_conv3 = use_extra_conv3
        if use_deformable:
            del self.content_encoder
            self.content_encoder = ModulatedDeformConv2dPack(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                stride=1,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
                deform_groups=1)
        self.use_encoder2 = use_encoder2
        del self.channel_compressor
        assert upsample_mode == 'nearest'
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.ss = SpatialSuppression(kernel_size=7, align_corners=align_corners, upsample_mode=upsample_mode)
        if self.use_leanable_hpf:
            self.LPF = LHPFConv3(channels=3, stride=1, padding=1)
            if self.use_extra_conv3:
                self.extra_conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=1, dilation=1, groups=1)
        self.init_weights()

    def _forward(self, feat, seg, img):
        if self.use_leanable_hpf:
            img_hf = self.LPF(img)
        else:
            lr_img = resize(
                input=img,
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            lr_img = resize(
                input=lr_img,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            img_hf = img - lr_img + img
        feat = resize(
            input=feat,
            size=img_hf.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        seg = resize(
            input=seg,
            size=img_hf.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # _ = self._do_nothing(seg)
        # feat_img = torch.cat([feat, img_hf], dim=1)
        # compressed_x = self.channel_compressor(feat)
        if self.use_extra_conv3:
            img_hf = self.extra_conv3(img_hf)
        mask = self.content_encoder(torch.cat([feat, img_hf], dim=1))
        mask = self.kernel_normalizer(mask)
        seg = self.feature_reassemble(seg, mask)
        feat = self.feature_reassemble(feat, mask)
        # print(seg.shape)
        return seg, feat

    def forward(self, feat, seg, img, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, feat, seg, img)
        else:
            return self._forward(feat, seg, img)

class CascadeImageGuideUpsample(nn.Module):
    def __init__(self, n=2, **kargs):
        super().__init__()
        self.n = n
        self.align_corners = kargs.get('align_corners', False)
        self.up = nn.ModuleList()
        for _ in range(n):
            self.up.append(ImageGuideUpsample(**kargs))

    def forward(self, feat, seg, img):
        for i in range(self.n):
            resized_img = resize(
                input=img,
                size=(2 * feat.size(-2), 2 * feat.size(-1)),
                mode='bilinear',
                align_corners=self.align_corners)
            # print(resized_img.shape)
            seg, feat = self.up[i](feat, seg, resized_img)
            # print(seg.shape, feat.shape)
        return seg

# class ImageFusion2(ImageGuideUpsample):
#     def __init__(self,
#                 num_classes=0,
#                 concat_imghf_for_seg=False,
#                 **kwargs):
#         super().__init__(**kwargs)
#         self.concat_imghf_for_seg = concat_imghf_for_seg
#         if concat_imghf_for_seg:
#             self.imghf_seg = nn.Conv2d(in_channels=3, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=False)

#     def forward(self, feat, seg, img):
#         seg, feat = super().forward(feat, seg, img)
#         if self.concat_imghf_for_seg:
#             seg += self.imghf_seg(img_hf)
#         return seg, feat
        
class ImageFusion(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor=1,
                up_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,   
                num_classes=0,
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                hr_residual=False,
                use_deformable=False,
                use_leanable_hpf=False,
                concat_imghf_for_seg=False,
                **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        # self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        # self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)

        self.hr_content_encoder = nn.Conv2d(
            hr_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.lr_content_encoder = nn.Conv2d(
            lr_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_spatial_residual = use_spatial_residual
        # self._do_nothing = nn.Conv2d(19, 1, 1)
        self.use_deformable = use_deformable
        self.use_leanable_hpf = use_leanable_hpf
        if use_deformable:
            del self.content_encoder
            self.content_encoder = ModulatedDeformConv2dPack(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                stride=1,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
                deform_groups=1)
        self.use_encoder2 = use_encoder2
        # del self.hr_channel_compressor
        # del self.lr_channel_compressor
        assert upsample_mode == 'nearest'
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.ss = SpatialSuppression(kernel_size=7, align_corners=align_corners, upsample_mode=upsample_mode)
        if self.use_leanable_hpf:
            self.LPF = LHPFConv3(channels=3, stride=1, padding=1)
        self.concat_imghf_for_seg = concat_imghf_for_seg
        if concat_imghf_for_seg:
            self.imghf_seg = nn.Conv2d(in_channels=3, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.hr_content_encoder, std=0.001)
        normal_init(self.lr_content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def carafe_forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        x = self.feature_reassemble(x, mask)
        return x

    def forward(self, feat, seg, img):
        if self.use_leanable_hpf:
            img_hf = self.LPF(img)
        else:
            lr_img = resize(
                input=img,
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            lr_img = resize(
                input=lr_img,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            img_hf = img - lr_img + img
        # feat = resize(
        #     input=feat,
        #     size=img_hf.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # seg = resize(
        #     input=seg,
        #     size=img_hf.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # _ = self._do_nothing(seg)
        # feat_img = torch.cat([feat, img_hf], dim=1)
        # compressed_x = self.channel_compressor(feat)
        # compressed_hr_feat = self.hr_channel_compressor(img_hf)
        # compressed_lr_feat = self.lr_channel_compressor(feat)
        mask = self.hr_content_encoder(img_hf) + F.interpolate(self.lr_content_encoder(feat), scale_factor=2)
        # mask = self.content_encoder(torch.cat([feat, img_hf], dim=1))
        mask = self.kernel_normalizer(mask)
        # seg = self.feature_reassemble(seg, mask)
        # feat = self.feature_reassemble(feat, mask)
        seg = carafe(seg, mask, self.up_kernel, self.up_group, 2)
        feat = carafe(feat, mask, self.up_kernel, self.up_group, 2)
        # print(seg.shape)
        if self.concat_imghf_for_seg:
            seg += self.imghf_seg(img_hf)
        return seg, feat

class CascadeImageFusion(nn.Module):
    def __init__(self, n=2, **kargs):
        super().__init__()
        self.n = n
        self.align_corners = kargs.get('align_corners', False)
        self.up = nn.ModuleList()
        for _ in range(n):
            self.up.append(ImageFusion(**kargs))

    def forward(self, feat, seg, img):
        for i in range(self.n):
            resized_img = resize(
                input=img,
                size=(2 * feat.size(-2), 2 * feat.size(-1)),
                mode='bilinear',
                align_corners=self.align_corners)
            # print(resized_img.shape)
            seg, feat = self.up[i](feat, seg, resized_img)
            # print(seg.shape, feat.shape)
        return seg

class FeatureAlignDyHPFusion2(DyHPFusion2):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                in_nc=128, 
                out_nc=128, 
                norm=None, 
                **kwargs):
        super().__init__(**kwargs)
        self.FA = FeatureAlign_V2(in_nc, out_nc, norm)

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            _, hr_feat, lr_feat = self.FA2(hr_feat, lr_feat)
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

class DyHPFusion(nn.Module):
    """
    Low-pass filter (dilation = 1) and local relation (dilation = 2, 4, 8 & etc) 
    can be implemented with the same function by using different dilations.

    """
    def __init__(self, in_channels, kernel_size=3, stride=1, groups=1, dilation=1, 
                use_adaptive_residual=False, use_channel=False, 
                # pad_mode='ada'
                # pad_mode='replicate',
                pad_mode='reflect',
                align_corners=False,
                upsample_mode='bilinear',
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.dilation = dilation
        self.pad = int((kernel_size-1) // 2 * self.dilation)
        self.use_channel = use_channel
        self.use_adaptive_residual = use_adaptive_residual
        self.pad_mode = pad_mode
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        # reflect replicate

        # self.weight_net = ConvModule(
        #     in_channels,
        #     groups * kernel_size ** 2,
        #     kernel_size,
        #     padding=self.pad,
        #     dilation=self.dilation,
        #     conv_cfg=None,
        #     # norm_cfg=dict(type='BN', requires_grad=True),
        #     norm_cfg=dict(type='SyncBN', requires_grad=True),
        #     # act_cfg=dict(type='ReLU')
        #     act_cfg=None
        # )
        if self.pad_mode == 'reflect':
            self.conv_pad = nn.ReflectionPad2d(padding=self.pad)  
        elif self.pad_mode == 'replicate':
            self.conv_pad = nn.ReplicationPad2d(padding=self.pad)
        elif self.pad_mode == 'ada':
            self.conv_pad = Adapad(padding=self.pad)
        else:
            raise NotImplementedError
        print(self.conv_pad)
        self.weight_net = nn.Sequential(
            # conv_pad,
            nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
                    kernel_size=kernel_size, bias=False, padding=0, groups=groups),
            # nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
            #         kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
            # nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            nn.SyncBatchNorm(self.groups * kernel_size ** 2), 
            # nn.ReLU(True)
            # nn.Softmax(dim=1)
        )
        self.weight_net2 = nn.Sequential(
            # conv_pad,
            nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
                    kernel_size=kernel_size, bias=False, padding=0, groups=groups),
            # nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
            #         kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
            # nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            nn.SyncBatchNorm(self.groups * kernel_size ** 2), 
            # nn.ReLU(True)
            # nn.Softmax(dim=1)
        )
        # self.weight_net[0].weight[:, :, 1, 1] = 1.

        if self.use_adaptive_residual:
            self.use_adaptive_residual = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            # nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
            # nn.BatchNorm2d(in_channels // 4), 
            # nn.SyncBatchNorm(in_channels // 4), 
            # nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels = 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4), 
                nn.SyncBatchNorm(in_channels // 4), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = groups * kernel_size ** 2, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.kaiming_normal_(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def _forward(self, hr_feat, lr_feat):
        """
        x for generate pixel relations
        """
        lr_feat = resize(
            input=lr_feat,
            size=hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        x = torch.cat([hr_feat, lr_feat], dim=1)
        _, _, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        # weight = self.weight_net(x) 
        # weight = self.weight_net(self.conv_pad(x)) 
        weight2 = self.weight_net2(self.conv_pad(x)) 
        if self.use_channel:
            weight = weight * self.channel_net(x)

        # pixel_relation = weight
        if hr_feat is not None:
            assert x.size(-1) == hr_feat.size(-1)
            assert x.size(-2) == hr_feat.size(-2)
            b, c, _, _ = hr_feat.shape

            # weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
            # weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

            weight2 = weight2.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
            weight2 = weight2.repeat(1, 1, c // self.groups, 1, 1, 1)

            if self.use_channel:
                tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
                # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
                # print(weight.shape)
                weight = weight * tmp

            # weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            # weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

            weight2 = weight2.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            weight2 = weight2.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
            # _weight_copy = weight[0, 0, 0, 0, 0, :, :].clone()
            # print(weight[0, 0, 0, 0, 0, :, :])
            # weight = torch.inverse(weight)
            # print(weight[0, 0, 0, 0, 0, :, :])
            # print(weight[0, 0, 0, 0, 0, :, :] @ _weight_copy)

            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode='reflect')
            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode=self.pad_mode)
            # pad_hr_feat = self.conv_pad(hr_feat)
            pad_lr_feat = self.conv_pad(lr_feat)
            # shape:  B x C x H // stride x W //stride x ksize x ksize
            # pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            # shape:  B x (C x ksize x ksize) x H // stride x W //stride
            # pad_hr_feat = F.unfold(pad_hr_feat, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride, dilation=self.dilation)
            pad_lr_feat = F.unfold(pad_lr_feat, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride, dilation=self.dilation)
            # print(pad_x.shape)
            # pad_hr_feat = pad_hr_feat.reshape(b, self.groups, c // self.groups, self.kernel_size, self.kernel_size, oh, ow).permute(0, 1, 2, 5, 6, 3, 4)
            pad_lr_feat = pad_lr_feat.reshape(b, self.groups, c // self.groups, self.kernel_size, self.kernel_size, oh, ow).permute(0, 1, 2, 5, 6, 3, 4)
            # pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
            # print(pad_x.shape)
            # hr_res = weight * pad_hr_feat
            # hr_res = hr_feat - hr_res.sum(dim=(-1, -2)).reshape(b, c, oh, ow) # high-pass

            lr_res = weight2 * pad_lr_feat
            lr_res = lr_res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)

            # return pixel_relation, hr_res, lr_res
            return weight2, hr_feat, lr_res
            # return pixel_relation, hr_res, lr_feat
        else:
            return pixel_relation

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

class SpatialPoolingDyHPFusion(nn.Module):
    """
    Low-pass filter (dilation = 1) and local relation (dilation = 2, 4, 8 & etc) 
    can be implemented with the same function by using different dilations.

    """
    def __init__(self, in_channels=4, kernel_size=3, stride=1, groups=1, dilation=1, 
                use_channel=False, 
                # pad_mode='ada'
                # pad_mode='replicate',
                pad_mode='reflect',
                align_corners=False,
                upsample_mode='bilinear',
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.dilation = dilation
        self.pad = int((kernel_size-1) // 2 * self.dilation)
        self.use_channel = use_channel
        # self.use_adaptive_residual = use_adaptive_residual
        self.pad_mode = pad_mode
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        # reflect replicate

        if self.pad_mode == 'reflect':
            self.conv_pad = nn.ReflectionPad2d(padding=self.pad)  
        elif self.pad_mode == 'replicate':
            self.conv_pad = nn.ReplicationPad2d(padding=self.pad)
        else:
            raise NotImplementedError
        print(self.conv_pad)
        self.weight_net = nn.Sequential(
            # conv_pad,
            nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
                    kernel_size=kernel_size, bias=False, padding=0, groups=groups),
            # nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
            #         kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
            # nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            nn.SyncBatchNorm(self.groups * kernel_size ** 2), 
        )
        self.weight_net2 = nn.Sequential(
            # conv_pad,
            nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
                    kernel_size=kernel_size, bias=False, padding=0, groups=groups),
            # nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
            #         kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
            # nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            nn.SyncBatchNorm(self.groups * kernel_size ** 2), 
            # nn.ReLU(True)
            # nn.Softmax(dim=1)
        )
        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4), 
                nn.SyncBatchNorm(in_channels // 4), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = groups * kernel_size ** 2, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.kaiming_normal_(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def get_max_avg_cat(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = dct.dct_2d(avg_out)
        # max_out = dct.dct_2d(avg_out)
        x = torch.cat([avg_out, max_out], dim=1)
        return x

    def _forward(self, hr_feat, lr_feat):
        """
        x for generate pixel relations
        """
        lr_feat = resize(
            input=lr_feat,
            size=hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        hr_feat_pooled = self.get_max_avg_cat(hr_feat)
        lr_feat_pooled = self.get_max_avg_cat(lr_feat)
        x = torch.cat([hr_feat_pooled, lr_feat_pooled], dim=1)
        _, _, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        # weight = self.weight_net(x) 
        weight = self.weight_net(self.conv_pad(x)) 
        weight2 = self.weight_net2(self.conv_pad(x)) 
        if self.use_channel:
            weight = weight * self.channel_net(x)

        pixel_relation = weight
        if hr_feat is not None:
            assert x.size(-1) == hr_feat.size(-1)
            assert x.size(-2) == hr_feat.size(-2)
            b, c, _, _ = hr_feat.shape

            weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
            weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

            weight2 = weight2.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
            weight2 = weight2.repeat(1, 1, c // self.groups, 1, 1, 1)

            if self.use_channel:
                tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
                # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
                # print(weight.shape)
                weight = weight * tmp

            weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

            weight2 = weight2.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            weight2 = weight2.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

            # _weight_copy = weight[0, 0, 0, 0, 0, :, :].clone()
            # print(weight[0, 0, 0, 0, 0, :, :])
            # weight = torch.inverse(weight)
            # print(weight[0, 0, 0, 0, 0, :, :])
            # print(weight[0, 0, 0, 0, 0, :, :] @ _weight_copy)

            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode='reflect')
            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode=self.pad_mode)
            pad_hr_feat = self.conv_pad(hr_feat)
            pad_lr_feat = self.conv_pad(lr_feat)
            # shape:  B x C x H // stride x W //stride x ksize x ksize
            # pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            # shape:  B x (C x ksize x ksize) x H // stride x W //stride
            pad_hr_feat = F.unfold(pad_hr_feat, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride, dilation=self.dilation)
            pad_lr_feat = F.unfold(pad_lr_feat, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride, dilation=self.dilation)
            # print(pad_x.shape)
            pad_hr_feat = pad_hr_feat.reshape(b, self.groups, c // self.groups, self.kernel_size, self.kernel_size, oh, ow).permute(0, 1, 2, 5, 6, 3, 4)
            pad_lr_feat = pad_lr_feat.reshape(b, self.groups, c // self.groups, self.kernel_size, self.kernel_size, oh, ow).permute(0, 1, 2, 5, 6, 3, 4)
            # pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
            # print(pad_x.shape)
            hr_res = weight * pad_hr_feat
            hr_res = hr_feat - hr_res.sum(dim=(-1, -2)).reshape(b, c, oh, ow) # high-pass

            # lr_res = weight2 * pad_lr_feat
            lr_res = weight * pad_lr_feat
            lr_res = lr_res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)

            return pixel_relation, hr_res, lr_res
            # return pixel_relation, hr_res, lr_feat
        else:
            return pixel_relation

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

class SpatialSuppression(nn.Module):
    def __init__(self, kernel_size=7, upsample_mode='bilinear', align_corners=False):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.sigmoid = nn.Sigmoid()

    def get_avg_max(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        # return self.sigmoid(x)
        return x

    def forward(self, hr_feat, lr_feat, residual=False):
        if lr_feat.size(-1) != hr_feat.size(-1):
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        hr = self.get_avg_max(hr_feat)
        lr = self.get_avg_max(lr_feat)
        att = self.conv1(torch.cat([hr, lr],  dim=1))
        att = self.sigmoid(att)
        if residual:
            return att, hr_feat * att + hr_feat, lr_feat
        else:
            return att, hr_feat * att, lr_feat


def c2_xavier_fill(module: nn.Module) -> None:
    # """
    # https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/weight_init.html
    # Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    # Also initializes `module.bias` to 0.

    # Args:
    #     module (torch.nn.Module): module to initialize.
    # """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm='GN'):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, dilation=1, kernel_size=1, bias=False),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=1, dilation=1, kernel_size=1, bias=False), #original FaPN
            # nn.GroupNorm(32, out_channels), 
        )
        self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, dilation=1, kernel_size=1, bias=False)
        self.conv = nn.Identity()
        # nn.init.xavier_uniform(self.conv_atten[0].weight)
        # c2_xavier_fill(self.conv_atten[0].weight)
        # nn.init.xavier_uniform(self.conv.weight)
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None, deform_groups=8):
        super().__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.offset = nn.Conv2d(in_channels=out_nc * 2, out_channels=out_nc, stride=1, dilation=1, kernel_size=1, bias=False)
        # self.dcpack_L2 = ModulatedDeformConv2d(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=deform_groups, extra_offset_mask=True)
        # self.dcpack_L2 = DCNv2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=deform_groups, extra_offset_mask=True)
        # https://github.com/EMI-Group/FaPN/blob/main/DCNv2/dcn_v2.py
        self.relu = nn.ReLU(inplace=True)
        # c2_xavier_fill(self.offset)
        # nn.init.xavier_uniform(self.offset.weight)
        # weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return offset, feat_arm, feat_align

class FeatureSelectionModuleOri(nn.Module):
    def __init__(self, in_channels, out_channels, norm='GN'):
        super().__init__()
        self.conv_atten = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=1, dilation=1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, dilation=1, kernel_size=1, bias=False)

        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)
        c2_xavier_fill(self.conv_atten)
        c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d as OriDCNv2
# from dcn_v2 import DCNv2 as OriDCNv2
# from dcn_v2 import dcn_v2_conv
class dcn_v2(nn.Module):
# class dcn_v2(OriDCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1, extra_offset_mask=False,):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        raise NotImplementedError
        self.extra_offset_mask = extra_offset_mask
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, main_path=None):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)       # each has self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] channels
        offset = torch.cat((o1, o2), dim=1)  # x, y [0-8]: the first group,
        # o1, o2 = o1.data.cpu().numpy(), o2.data.cpu().numpy()
        # o = o1[0]   # first image in the batch
        # print(o1[0])
        # return
        # k = 0
        # img_h, img_w = 128, 256
        # img_r, img_c, inter = 3, 3, 5
        # img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        # to_image = Image.new('RGB', img_size, 'white')
        # print()
        # for j in range(o.shape[0]):     # for group
        #     if j % 9 == 0 and j != 0:  # different kernel
        #         dst_file = os.path.join(main_path, 'offset_{}to{}.png'.format(j-9, j))
        #         # save first
        #         to_image.save(dst_file)
        #         # new image
        #         img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        #         to_image = Image.new('RGB', img_size, 'white')
        #     feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        #     for x, y in range
        #     feature_img = Image.fromarray(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
        #     index_r, index_c = j // img_c, j % img_c
        #     to_image.paste(feature_img, (index_c * (img_w + inter), index_r * (img_h + inter)))
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups,)
    
from mmseg.models.backbones.interp2d import ConvSaliencySampler, ModulatedDeformConv2d, modulated_deform_conv2d
class dcnv2_AS(OriDCNv2):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1, extra_offset_mask=False, 
                 only_center=True, radius=2):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
    # def __init__(self, *args, only_center=True, group=1, radius=8, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.CA = ChannelAttention(in_planes=self.in_channels)
        # self.SA = SpatialAttention(kernel_size=7)
        self.extra_offset_mask = extra_offset_mask
        self.only_center = only_center
        self.radius = radius
        # self.group = group
        self.deform_groups = deformable_groups
        if self.only_center:
            self.AS_offset = ConvSaliencySampler(
                kernel_size=self.kernel_size[0],
                c_out=deformable_groups,
                # group=group,
                feat_dim=self.in_channels , 
                stride=self.stride, 
                # stride=1, 
                padding_size=self.radius, fwhm=self.radius, dilation=1, use_checkpoint=True, detach=False, align_corners=False,
                sampler_mode='conv3', 
                # padding_mode='reflect', # better than replicate
                coord_mode='yx',
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
                c_out=deformable_groups * self.kernel_size[0] * self.kernel_size[1],
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
        # self.bn = nn.SyncBatchNorm(self.in_channels)
        # self.relu = nn.ReLU(True)
        self.init_weights()
        # self.AS_offset.filter.weight.requires_grad = False

    def init_weights(self):
        # super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

        if hasattr(self, 'conv_mask'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    def forward(self, feat, main_path=None):
        if self.extra_offset_mask:
            out = feat[0]
            x = feat[1] # for offset
            # x = self.relu(self.bn(x))
        else:
            # out = feat
            raise NotImplementedError
        # x = self.CA(x) * x
        # x = self.SA(x) * x
        # /data3/chenlinwei/code/ResolutionDet/detectron2/DCNv2-pytorch_1.9/src/cpu/dcn_v2_im2col_cpu.cpp
        # const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        """
        # /data3/chenlinwei/code/ResolutionDet/detectron2/DCNv2-pytorch_1.9/src/cpu/dcn_v2_im2col_cpu.cpp
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

            # AS_offset = self.AS_offset(self.relu(self.bn(x)))
            # AS_offset = self.AS_offset(self.bn(x))
            AS_offset = self.AS_offset(x)
            # print(AS_offset.shape)
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
            offset = torch.cat([*offset_list[:o_n // 2], AS_offset, *offset_list[-(o_n // 2):]], 2).reshape(b, -1, h, w)
        else:
            offset = self.AS_offset(x)

        # print(offset.shape)
        # print(x.shape)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        # print(mask.shape)
        # return modulated_deform_conv2d(out, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, 1,
        #                                self.deform_groups)
        return dcn_v2_conv(out, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)
    
class FeatureAlignOri(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None, deform_groups=8, use_adaptive_sampling=False):
        super().__init__()
        self.lateral_conv = FeatureSelectionModuleOri(in_nc, out_nc, norm="")
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.offset = nn.Conv2d(in_channels=out_nc * 2, out_channels=out_nc, stride=1, dilation=1, kernel_size=1, bias=False)
        # self.dcpack_L2 = ModulatedDeformConv2d(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        # self.dcpack_L2 = DCNv2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=deform_groups, extra_offset_mask=True)
        if use_adaptive_sampling:
            self.dcpack_L2 = dcnv2_AS(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=deform_groups, extra_offset_mask=True)
        else:
            self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=deform_groups, extra_offset_mask=True)
        # https://github.com/EMI-Group/FaPN/blob/main/DCNv2/dcn_v2.py
        self.relu = nn.ReLU(inplace=True)
        c2_xavier_fill(self.offset)
        # weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return offset, feat_arm, feat_align

    
class FreqAwareFeatureAlignOri(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None, deform_groups=8, use_checkpoint=True):
        super().__init__()
        self.lateral_conv = FeatureSelectionModuleOri(in_nc, out_nc, norm="")
        # self.freqfusion = DyHPFusion2(channels=out_nc * 2, scale_factor=1, up_kernel=3, up_group=1, 
        #             upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
        #             use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
        #             use_high_pass=True, use_low_pass=True)
        # self.freqfusion = FreqFusion(hr_channels=out_nc, lr_channels=out_nc, scale_factor=1, up_kernel=3, up_group=1, 
        #             upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
        #             use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
        #             use_high_pass=True, use_low_pass=True)
        # self.freqfusion = FreqFusionFastEval(hr_channels=out_nc, lr_channels=out_nc, channels=out_nc * 2, scale_factor=1, up_kernel=3, up_group=1, 
        #             upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
        #             use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
        #             use_high_pass=True, use_low_pass=True)
        self.freqfusion = FreqFusion2encoder(hr_channels=out_nc, lr_channels=out_nc,                
                                             scale_factor=1,
                                            lowpass_kernel=5,
                                            highpass_kernel=3,
                                            up_group=1,
                                            encoder_kernel=3,
                                            encoder_dilation=1,
                                            compressed_channels=64,        
                                            align_corners=False,
                                            upsample_mode='nearest',
                                            # use_encoder2=False,
                                            use_spatial_suppression=False,
                                            use_spatial_residual=False,
                                            use_high_pass=True,
                                            use_low_pass=True,
                                            hr_residual=True,
                                            semi_conv=True,)
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.offset = nn.Conv2d(in_channels=out_nc * 2, out_channels=out_nc, stride=1, dilation=1, kernel_size=1, bias=False)
        # self.dcpack_L2 = ModulatedDeformConv2d(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=deform_groups, extra_offset_mask=True)
        # self.dcpack_L2 = DCNv2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=deform_groups, extra_offset_mask=True)
        # https://github.com/EMI-Group/FaPN/blob/main/DCNv2/dcn_v2.py
        self.relu = nn.ReLU(inplace=True)
        self.use_checkpoint = use_checkpoint
        c2_xavier_fill(self.offset)
        # weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if self.use_checkpoint:
            feat_arm = checkpoint(self.lateral_conv, feat_l)  # 0~1 * feats
            _, feat_arm, feat_up = self.freqfusion(hr_feat=feat_arm, lr_feat=feat_s, use_checkpoint=True)
        else:
            feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
            _, feat_arm, feat_up = self.freqfusion(hr_feat=feat_arm, lr_feat=feat_s)
        # if feat_l.size()[2:] != feat_s.size()[2:]:
        #     feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        # else:
        #     feat_up = feat_s
        feat_arm += feat_up
        # feat_up += feat_arm_hi
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return offset, feat_arm, feat_align

class DCNv2(ModulatedDeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deform_groups=1, extra_offset_mask=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deform_groups)
        self.extra_offset_mask = extra_offset_mask
        self.deform_groups = deform_groups
        channels_ = self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, offset_bias=None):
        # input[]
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)       # each has self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] channels
        offset = torch.cat((o1, o2), dim=1)  # x, y [0-8]: the first group,
        # print('offset', offset)
        if offset_bias is not None: 
            offset += offset_bias
        # o1, o2 = o1.data.cpu().numpy(), o2.data.cpu().numpy()
        # o = o1[0]   # first image in the batch
        # print(o1[0])
        # return
        # k = 0
        # img_h, img_w = 128, 256
        # img_r, img_c, inter = 3, 3, 5
        # img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        # to_image = Image.new('RGB', img_size, 'white')
        # print()
        # for j in range(o.shape[0]):     # for group
        #     if j % 9 == 0 and j != 0:  # different kernel
        #         dst_file = os.path.join(main_path, 'offset_{}to{}.png'.format(j-9, j))
        #         # save first
        #         to_image.save(dst_file)
        #         # new image
        #         img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        #         to_image = Image.new('RGB', img_size, 'white')
        #     feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        #     for x, y in range
        #     feature_img = Image.fromarray(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
        #     index_r, index_c = j // img_c, j % img_c
        #     to_image.paste(feature_img, (index_c * (img_w + inter), index_r * (img_h + inter)))
        mask = torch.sigmoid(mask)
        return super().forward(input, offset=offset, mask=mask)