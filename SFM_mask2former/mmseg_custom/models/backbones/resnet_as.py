# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
from mmseg.models.backbones.resnet import ResNet, ResNetV1c, Bottleneck
from mmseg.registry import MODELS
# from mmengine.registry import MODELS
from torch.utils.tensorboard import SummaryWriter
from mmseg_custom.models.backbones.interp2d import SaliencySampler
from torch.utils.checkpoint import checkpoint

@MODELS.register_module()
class ResNetSaliencySampler(ResNet):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    def __init__(self, log_dir=None,  
        sampler_num=3,
        sampler_strides=(1, 1, 1), 
        sampler_paddings=(5, 5, 5), 
        sampler_fwhm=(3, 3, 3),  
        sampler_mode=('avgpsp_semantic_edge', 'avgpsp_semantic_edge', 'avgpsp_semantic_edge'),  
        align_corners=True,
        psp_ratio=4,
        use_avg=(True, True, True), 
        use_checkpoint=True,
        use_lpf=False,
        lpf_kernel=3,
        lpf_group=1,
        lpf_dilation=1,
        lpf_mode = 'feat',
        **kwargs):
        # super().__init__(deep_stem=True, avg_down=True, **kwargs)
        super().__init__(**kwargs)
        if kwargs['depth'] < 50:
            sampler_channels = [64, 128, 256]
        else:
            sampler_channels = [256, 512, 1024]
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.sampler_num = sampler_num
        self.use_checkpoint = use_checkpoint
        self.use_lpf = use_lpf
        self.lpf_group = lpf_group
        self.lpf_kernel = lpf_kernel
        self.lpf_dilation = lpf_dilation
        self.SS = nn.ModuleList([
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SaliencySampler(feat_dim=sampler_channels[0], stride=sampler_strides[0], padding_size=sampler_paddings[0], fwhm=sampler_fwhm[0], sampler_mode=sampler_mode[0], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=3 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[1], stride=sampler_strides[1], padding_size=sampler_paddings[1], fwhm=sampler_fwhm[1], sampler_mode=sampler_mode[1], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=2 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[2], stride=sampler_strides[2], padding_size=sampler_paddings[2], fwhm=sampler_fwhm[2], sampler_mode=sampler_mode[2], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=1 else nn.Identity(),
        ])
        if use_lpf:
            self.lpf_mode = lpf_mode
            if ('feat' in lpf_mode) and ('offset' in lpf_mode):
                LPF_channels = [i + 4 for i in sampler_channels]
            elif ('feat' in lpf_mode):
                LPF_channels = [i for i in sampler_channels]
            elif ('offset' in lpf_mode):
                LPF_channels = [4 for i in sampler_channels]
            else:
                raise NotImplementedError
            self.LPF = nn.ModuleList([
                nn.Identity() ,
                LocalPixelRelationConv(LPF_channels[0], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                LocalPixelRelationConv(LPF_channels[1], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                LocalPixelRelationConv(LPF_channels[2], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                # Downsample_PASA_group_softmax(LPF_channels[0], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                # Downsample_PASA_group_softmax(LPF_channels[1], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                # Downsample_PASA_group_softmax(LPF_channels[2], kernel_size=lpf_kernel, stride=1, group=lpf_group),
            ])

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            # x = self.stem(x)
            if self.use_checkpoint: x.requires_grad = True
            x = checkpoint(self.stem, x) if self.use_checkpoint else self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        reversed_outs = []
        grids = []
        smaps = []
        x_sampleds = []
        # maxpool -> stage1 (4x) -> SS1, stage2 (8x) -> SS2, stage3 (16x) -> SS3, stage4 (32x) -> seghead
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 0:
                x = self.SS[i](x) # maxpooling
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x)
                else:
                    x = res_layer(x) 
                grid = None
                saliency = None
                x_sampled = None
                _x_sampled = None
            elif i in (1, 2, 3)[-self.sampler_num:]:
                # _, _, h, w = x.shape
                if self.training:
                    _, _x_sampled, _, _ = self.SS[i](x.detach(), sample_with_grid=True)
                x, x_sampled, saliency, grid = self.SS[i](x, sample_with_grid=True)
                if self.use_lpf:
                    if self.lpf_mode == 'flc':
                        x_sampled = self.LPF[i](x)
                    else:
                        offset = self.SS[i].get_offset(grid)
                        if ('feat' in self.lpf_mode) and ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](torch.cat([x_sampled, offset.detach()], dim=1), x_sampled, use_checkpoint=True)
                        elif ('feat' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](x_sampled, x_sampled, use_checkpoint=True)
                        elif ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](offset, x_sampled, use_checkpoint=True)
                        else:
                            raise NotImplementedError
                # x_sampled = self.SS[i].sample_with_grid(x, grid)
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x_sampled)
                else:
                    x = res_layer(x_sampled)  
            else:
                x = res_layer(x)  
                grid = None
                saliency = None
                # raise NotImplementedError         

            if i in self.out_indices:
                # print(i)
                grids.append(grid)
                smaps.append(saliency)
                outs.append(x)
                x_sampleds.append(_x_sampled)
                # reversed_outs.append(x)

        # return tuple(outs)
        # return tuple(outs), tuple(grids), tuple(smaps)
        return outs, grids, {'smaps':smaps, 'x_sampleds':x_sampleds}
    
@MODELS.register_module()
class ResNetV1cSaliencySampler(ResNetV1c):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    def __init__(self, log_dir=None,  
        sampler_num=3,
        sampler_strides=(1, 1, 1), 
        sampler_paddings=(5, 5, 5), 
        sampler_fwhm=(3, 3, 3),  
        sampler_mode=('avgpsp_semantic_edge', 'avgpsp_semantic_edge', 'avgpsp_semantic_edge'),  
        align_corners=True,
        psp_ratio=4,
        use_avg=(True, True, True), 
        use_checkpoint=True,
        use_lpf=False,
        lpf_kernel=3,
        lpf_group=1,
        lpf_dilation=1,
        lpf_mode = 'feat',
        **kwargs):
        # super().__init__(deep_stem=True, avg_down=True, **kwargs)
        super().__init__(**kwargs)
        if kwargs['depth'] < 50:
            sampler_channels = [64, 128, 256]
        else:
            sampler_channels = [256, 512, 1024]
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.sampler_num = sampler_num
        self.use_checkpoint = use_checkpoint
        self.use_lpf = use_lpf
        self.lpf_group = lpf_group
        self.lpf_kernel = lpf_kernel
        self.lpf_dilation = lpf_dilation
        self.SS = nn.ModuleList([
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SaliencySampler(feat_dim=sampler_channels[0], stride=sampler_strides[0], padding_size=sampler_paddings[0], fwhm=sampler_fwhm[0], sampler_mode=sampler_mode[0], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=3 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[1], stride=sampler_strides[1], padding_size=sampler_paddings[1], fwhm=sampler_fwhm[1], sampler_mode=sampler_mode[1], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=2 else nn.Identity(),
            SaliencySampler(feat_dim=sampler_channels[2], stride=sampler_strides[2], padding_size=sampler_paddings[2], fwhm=sampler_fwhm[2], sampler_mode=sampler_mode[2], log_dir=self.log_dir, align_corners=align_corners, psp_ratio=psp_ratio) if sampler_num >=1 else nn.Identity(),
        ])
        if use_lpf:
            self.lpf_mode = lpf_mode
            if ('feat' in lpf_mode) and ('offset' in lpf_mode):
                LPF_channels = [i + 4 for i in sampler_channels]
            elif ('feat' in lpf_mode):
                LPF_channels = [i for i in sampler_channels]
            elif ('offset' in lpf_mode):
                LPF_channels = [4 for i in sampler_channels]
            else:
                raise NotImplementedError
            self.LPF = nn.ModuleList([
                nn.Identity() ,
                LocalPixelRelationConv(LPF_channels[0], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                LocalPixelRelationConv(LPF_channels[1], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                LocalPixelRelationConv(LPF_channels[2], kernel_size=lpf_kernel, stride=1, groups=lpf_group, dilation=lpf_dilation),
                # Downsample_PASA_group_softmax(LPF_channels[0], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                # Downsample_PASA_group_softmax(LPF_channels[1], kernel_size=lpf_kernel, stride=1, group=lpf_group),
                # Downsample_PASA_group_softmax(LPF_channels[2], kernel_size=lpf_kernel, stride=1, group=lpf_group),
            ])

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            # x = self.stem(x)
            if self.use_checkpoint: x.requires_grad = True
            x = checkpoint(self.stem, x) if self.use_checkpoint else self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        reversed_outs = []
        grids = []
        smaps = []
        x_sampleds = []
        # maxpool -> stage1 (4x) -> SS1, stage2 (8x) -> SS2, stage3 (16x) -> SS3, stage4 (32x) -> seghead
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 0:
                x = self.SS[i](x) # maxpooling
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x)
                else:
                    x = res_layer(x) 
                grid = None
                saliency = None
                x_sampled = None
                _x_sampled = None
            elif i in (1, 2, 3)[-self.sampler_num:]:
                # _, _, h, w = x.shape
                if self.training:
                    _, _x_sampled, _, _ = self.SS[i](x.detach(), sample_with_grid=True)
                x, x_sampled, saliency, grid = self.SS[i](x, sample_with_grid=True)
                if self.use_lpf:
                    if self.lpf_mode == 'flc':
                        x_sampled = self.LPF[i](x)
                    else:
                        offset = self.SS[i].get_offset(grid)
                        if ('feat' in self.lpf_mode) and ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](torch.cat([x_sampled, offset.detach()], dim=1), x_sampled, use_checkpoint=True)
                        elif ('feat' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](x_sampled, x_sampled, use_checkpoint=True)
                        elif ('offset' in self.lpf_mode):
                            _, x_sampled = self.LPF[i](offset, x_sampled, use_checkpoint=True)
                        else:
                            raise NotImplementedError
                # x_sampled = self.SS[i].sample_with_grid(x, grid)
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x_sampled)
                else:
                    x = res_layer(x_sampled)  
            else:
                x = res_layer(x)  
                grid = None
                saliency = None
                # raise NotImplementedError         

            if i in self.out_indices:
                # print(i)
                grids.append(grid)
                smaps.append(saliency)
                outs.append(x)
                x_sampleds.append(_x_sampled)
                # reversed_outs.append(x)

        # return tuple(outs)
        # return tuple(outs), tuple(grids), tuple(smaps)
        return outs, grids, {'smaps':smaps, 'x_sampleds':x_sampleds}

@MODELS.register_module()
class ResNetV1cSaliencySamplerVis(ResNetV1cSaliencySampler):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            # x = self.stem(x)
            if self.use_checkpoint: x.requires_grad = True
            x = checkpoint(self.stem, x) if self.use_checkpoint else self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        reversed_outs = []
        grids = []
        smaps = {}
        # maxpool -> stage1 (4x) -> SS1, stage2 (8x) -> SS2, stage3 (16x) -> SS3, stage4 (32x) -> seghead
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 0:
                x = self.SS[i](x) # maxpooling
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x)
                else:
                    x = res_layer(x) 
                grid = None
                saliency = None
            elif i in (1, 2, 3)[-self.sampler_num:]:
                _, _, h, w = x.shape
                x, x_sampled, saliency, grid = self.SS[i](x, sample_with_grid=True)

                smaps['x'] = x
                smaps['grid'] = grid
                smaps['sampled'] = x_sampled
                smaps['adaptive'] = F.interpolate(x_sampled, size=(h // 2 + h % 2, w // 2 + w % 2), mode='bilinear', align_corners=(h % 2 == 1))
                smaps['uniform'] = x[:, :, ::2, ::2]
                # smaps['adaptive'] = x_sampled[:, :, ::2, ::2]
                # smaps['uniform'] = x[:, :, ::2, ::2]
                smaps['adaptive_up'] = self.SS[i].reverse(x, grid, h, w)
                smaps['uniform_up'] = F.interpolate(smaps['uniform'], size=(h, w), mode='bilinear', align_corners=(h % 2 == 1))
                if self.use_lpf:
                    offset = self.SS[i].get_offset(grid)
                    if ('feat' in self.lpf_mode) and ('offset' in self.lpf_mode):
                        _, x_sampled = self.LPF[i](torch.cat([x_sampled, offset.detach()], dim=1), x_sampled, use_checkpoint=True)
                    elif ('feat' in self.lpf_mode):
                        _, x_sampled = self.LPF[i](x_sampled, x_sampled, use_checkpoint=True)
                    elif ('offset' in self.lpf_mode):
                        _, x_sampled = self.LPF[i](offset, x_sampled, use_checkpoint=True)
                    else:
                        raise NotImplementedError
                # x_sampled = self.SS[i].sample_with_grid(x, grid)
                if self.use_checkpoint:
                    x = checkpoint(res_layer, x_sampled)
                else:
                    x = res_layer(x_sampled)  
            else:
                x = res_layer(x)  
                grid = None
                saliency = None
                # raise NotImplementedError         

            if i in self.out_indices:
                # print(i)
                grids.append(grid)
                # smaps.append(saliency)
                outs.append(x)
                # reversed_outs.append(x)
        # return tuple(outs)
        # return tuple(outs), tuple(grids), tuple(smaps)
        return outs, grids, smaps

class SpatialMaxAvgPooling(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        # return self.sigmoid(x)
        return x

class ASBottleneck(Bottleneck):
    def __init__(self, *args, 
                 align_corners=False,
                 use_grid_inv=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.align_corners = align_corners
        self.AS = SaliencySampler(feat_dim=self.inplanes, 
                                  stride=1, 
                                  use_checkpoint=False,
                                  padding_size=4, fwhm=3, sampler_mode='avg_lhpf_conv', detach=False,
                                  log_dir=None, align_corners=align_corners, psp_ratio=4)
        self.use_grid_inv = use_grid_inv
        if not self.use_grid_inv:
            # self.flow_conv = nn.Conv2d(self.inplanes + self.planes, 2, kernel_size=3, stride=1 if self.downsample == None else 2,
                                    #    padding=1, bias=False)
            self.offset2flow_conv = nn.Conv2d(2, 2, kernel_size=7, stride=1 if self.downsample == None else 2,
                                       padding=3, bias=False)
            self.feat2flow_conv = nn.Conv2d(4, 2, kernel_size=7, stride=1,
                                       padding=3, bias=False)
            self.spatialpool = SpatialMaxAvgPooling()
            self.feat2flow_conv.weight.data.zero_()
            
    def flow_warp(self, input, flow, size=None):
        # out_h, out_w = size
        n, c, h, w = input.size()
        out_h, out_w = h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, padding_mode='border')
        return output
    
    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)

            feature, x, saliency, grid = self.AS(x)

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            # if not self.use_grid_inv:
                # flow = self.flow_conv(torch.cat([identity, out], dim=1))
                # offset = self.AS.get_offset(grid=grid, cell_encode=False)
                # pooled_identity = self.spatialpool(identity)
                # pooled_out = self.spatialpool(out)
                # flow = self.flow_conv(torch.cat([offset, pooled_identity, pooled_out], dim=1))
                # out = self.flow_warp(out, flow)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.use_grid_inv:
                grid = F.interpolate(grid, identity.shape[2:], mode='bilinear', align_corners=self.align_corners)
                out = self.AS.reverse_with_inv_grid(grid=grid, sampled_feat=out, h=identity.shape[-2], w=identity.shape[-1])
            else:
                offset = self.AS.get_offset(grid=grid, cell_encode=False)
                pooled_identity = self.spatialpool(identity)
                pooled_out = self.spatialpool(out)
                offset2flow = self.offset2flow_conv(offset)
                feat2flow = self.feat2flow_conv(torch.cat([pooled_identity, pooled_out], dim=1))
                flow = offset2flow + feat2flow
                out = self.flow_warp(out, flow)
                pass

            out += identity
            
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
    

@MODELS.register_module()
class ASResNetV1c(ResNetV1c):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    arch_settings = {
        50: (ASBottleneck, (3, 4, 6, 3)),
        101: (ASBottleneck, (3, 4, 23, 3)),
        152: (ASBottleneck, (3, 8, 36, 3))
    }
    def __init__(self, 
                 align_corners=False,
                 normal_stage=2,
                 **kwargs):
        super().__init__(**kwargs)
        del self.res_layers
        self.res_layers = []
        inplanes = self.inplanes
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                # block=self.block if (i + 1) > self.frozen_stages else Bottleneck,
                block=self.block if (i + 1) > max(normal_stage, self.frozen_stages) else Bottleneck,
                align_corners=align_corners,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=self.block_init_cfg)
            inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        # print(self)
        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)