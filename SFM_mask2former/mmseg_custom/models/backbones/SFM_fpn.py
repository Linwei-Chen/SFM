# For FPN structure, 由于固定的 Barycentric 上采样在分辨率略高的情况下需要计算大量的三角形区域，非常慢，
# 因此使用Learnable Offset来构建三角形跳过三角形的构建与分配，然后直接使用 Barycentric 插值从non-uniform得到uniform的特征。并使用carafe算子节省显存实现LPRM

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def compute_real_grids(relative_grids: list, base_feature_shape: tuple, align_corners: bool) -> list:
    """
    根据相对采样网格列表，计算每个特征层对应的真实累积网格。

    Args:
        relative_grids (list): 包含相对采样网格的列表，例如 [grid_1, grid_2, grid_3]。
                               grid_1 用于 x1->x2, grid_2 用于 x2->x3, etc.
        base_feature_shape (tuple): 最高分辨率特征 x1 的形状 (B, C, H, W)。

    Returns:
        list: 包含每个尺度真实绝对坐标的列表 [real_grid_1, real_grid_2, ...]。
              real_grid_n 的形状为 [B, H_n, W_n, 2]。
    """
    B, _, H, W = base_feature_shape
    device = relative_grids[1].device
    
    # x1的真实网格就是其自身的均匀网格
    if align_corners:
        # y_coords = torch.linspace(-1, 1, 10)
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
    else:
        y_coords = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
        x_coords = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)

    # print([i.min() if i is not None else False for i in relative_grids])

    # if self.align_corners:
    #     x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / (w - 1.0)
    #     y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / (h - 1.0)
    # else:
    #     x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / w + 0.5 / w
    #     y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / h + 0.5 / h
    # coord_grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0)

    # real_grid_1 形状: [B, H, W, 2]
    real_grid_1 = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    real_grids = []
    current_real_grid = real_grid_1
    # print(len(relative_grids))

    for rel_grid in relative_grids:
        
        if rel_grid is None: 
            # current_real_grid = next_real_grid
            next_real_grid = current_real_grid
            real_grids.append(next_real_grid)
        else:
            # print(rel_grid.shape)
            # print(current_real_grid.shape)
            # rel_grid 形状: [B, H_in, W_in, 2]
            # current_real_grid 形状: [B, H_in, W_in, 2]
            # F.grid_sample 需要输入一个特征图和一个采样网格。
            # 这里我们巧妙地用它来对“坐标本身”进行重采样，从而实现网格变换的复合。
            # 我们对 current_real_grid (作为特征图) 使用 rel_grid (作为采样网格) 进行采样。
            next_real_grid = F.grid_sample(
                current_real_grid.permute(0, 3, 1, 2), # 将坐标维度(2)放到通道维
                rel_grid.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border',
                # align_corners=current_real_grid.size(2) // 2 == 1
                align_corners=align_corners
            ).permute(0, 2, 3, 1) # 再转换回来
            real_grids.append(next_real_grid)
            current_real_grid = next_real_grid
    
    # print(len(real_grids))
    return real_grids

# ==============================================================================
# 2. 全新修正的核心插值模块
# ==============================================================================

def _perform_learnabel_non_uniform_approx_interpolation_with_real_grid(
    feature_to_sample: torch.Tensor,
    offset_predictions: torch.Tensor,
    real_grid_of_feature: torch.Tensor,
    target_h: int,
    target_w: int,
    k: int = 3,
    power: int = 2,
    align_corners: bool = False,
    mode: str = 'barycentric_prox',
    centralized = False
) -> torch.Tensor:
    """
    修正后的、处理累积非均匀性的插值模块，支持'idw'和'barycentric'两种权重模式。
    Args:
        feature_to_sample (torch.Tensor): 需要采样的源特征图 (例如 x_n)。
        offset_predictions (torch.Tensor): 在目标均匀网格上预测的偏移量 [B, k*2, H_target, W_target]。
        real_grid_of_feature (torch.Tensor): 源特征图的真实（绝对）网格坐标 [B, H_src, W_src, 2]。
        target_h, target_w (int): 目标输出特征图的尺寸。
        k, power (int): IDW插值参数。
        align_corners (bool): F.grid_sample 的 align_corners 参数。

    Returns:
        torch.Tensor: 插值后生成的对齐特征图。
    """
    B, C, H_src, W_src = feature_to_sample.shape
    device = feature_to_sample.device

    # --- 步骤 1: 创建目标均匀网格 ---
    if align_corners:
        # y_coords = torch.linspace(-1, 1, 10)
        y_coords = torch.linspace(-1, 1, target_h, device=device)
        x_coords = torch.linspace(-1, 1, target_w, device=device)
    else:
        y_coords = torch.linspace(-1 + 1 / target_h, 1 - 1 / target_h, target_h, device=device)
        x_coords = torch.linspace(-1 + 1 / target_w, 1 - 1 / target_w, target_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
    # target_uniform_grid 形状: [B, H_target, W_target, 2]
    target_uniform_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # --- 步骤 2 (核心修正): 对像素偏移量进行正确的归一化 ---
    # offsets_pixel 形状: [B, H_target, W_target, k, 2]
    offsets_pixel = offset_predictions.view(B, k, 2, target_h, target_w).permute(0, 3, 4, 1, 2)
    if centralized:
        offsets_pixel = offsets_pixel - offsets_pixel.mean(dim=-2, keepdim=True) # center norm
    
    # 创建归一化张量，将像素偏移转换为[-1, 1]范围内的坐标偏移
    # 1个像素的偏移对应于 2/W 或 2/H 的归一化坐标变化
    norm = torch.tensor([target_w / 2, target_h / 2], device=device).view(1, 1, 1, 1, 2)
    offsets = offsets_pixel / norm
    
    # uniform_sample_locations 形状: [B, H_target, W_target, k, 2]
    uniform_sample_locations = target_uniform_grid.unsqueeze(-2) + offsets
    if align_corners:
        uniform_sample_locations = torch.clip(uniform_sample_locations, min=-1, max=1.0)
    else:
        uniform_sample_locations = torch.clip(uniform_sample_locations, min=-1, max=1.0)
        # uniform_sample_locations[..., 0] = torch.clip(uniform_sample_locations[..., 0], min=-1 + 1 / target_w, max=1 - 1 / target_w)
        # uniform_sample_locations[..., 1] = torch.clip(uniform_sample_locations[..., 1], min=-1 + 1 / target_h, max=1 - 1 / target_h)

    # --- 步骤 3 (修正): 对非均匀特征图进行采样 ---
    # 使用 `uniform_sample_locations` 来采样 `feature_to_sample`，因为 `feature_to_sample`
    # 本身是一个结构化的张量，可以用看似均匀的坐标进行采样。
    sampled_features_list = []
    for i in range(k):
        # current_uniform_grid 形状: [B, H_target, W_target, 2]
        current_uniform_grid = uniform_sample_locations[:, :, :, i, :]
        sampled_feat = F.grid_sample(
            feature_to_sample,
            current_uniform_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners
        )
        sampled_features_list.append(sampled_feat)

    # --- 步骤 4: 计算采样点在真实物理空间中的坐标 ---
    # if real_grid_of_feature is None:
    #     all_final_locations = uniform_sample_locations
    # else:
    final_sample_locations_list = []
    for i in range(k):
        # 使用 `uniform_sample_locations` 去采样 `real_grid_of_feature`
        # 将均匀空间的采样点“翻译”成非均匀空间中的真实坐标
        final_loc = F.grid_sample(
            real_grid_of_feature.permute(0, 3, 1, 2), # real_grid 需要 [B, 2, H_src, W_src]
            uniform_sample_locations[:, :, :, i, :], # grid 需要 [B, H_target, W_target, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners
        ).permute(0, 2, 3, 1) # 转换回 [B, H_target, W_target, 2]
        final_sample_locations_list.append(final_loc)
    
    # all_final_locations 形状: [B, H_target, W_target, k, 2]
    all_final_locations = torch.stack(final_sample_locations_list, dim=3)

    # --- 步骤 5: 根据模式计算权重 ---
    if mode == 'barycentric_prox':
        real_offsets = all_final_locations - target_uniform_grid.unsqueeze(-2)
        dists = torch.norm(real_offsets, dim=-1, p=2)
        weights = 1.0 / (dists.pow(power) + 1e-8)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        # weights 形状: [B, k, 1, H_target, W_target]
        weights = weights.permute(0, 3, 1, 2).unsqueeze(2)

    elif mode == 'barycentric':
        assert k == 3, "重心坐标插值模式 (barycentric) 必须使用 k=3 个邻居点。"
        
        # P: 目标点, P0, P1, P2: 三个邻居点
        p0 = all_final_locations[..., 0, :] # 形状: [B, H, W, 2]
        p1 = all_final_locations[..., 1, :]
        p2 = all_final_locations[..., 2, :]
        p = target_uniform_grid

        # 计算向量
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p - p0
        
        # 计算点积
        d00 = (v0 * v0).sum(dim=-1)
        d01 = (v0 * v1).sum(dim=-1)
        d11 = (v1 * v1).sum(dim=-1)
        d20 = (v2 * v0).sum(dim=-1)
        d21 = (v2 * v1).sum(dim=-1)
        
        # 计算分母，并添加epsilon防止除以零
        denom = d00 * d11 - d01 * d01
        
        # 计算权重 w1 和 w2
        w1 = (d11 * d20 - d01 * d21) / (denom + 1e-8)
        w2 = (d00 * d21 - d01 * d20) / (denom + 1e-8)

        # w1 = F.relu(w1, inplace=True)
        # w2 = F.relu(w2, inplace=True)
        w1 = torch.clip(w1, min=0, max=1.0)
        w2 = torch.clip(w2, min=0, max=1.0)
        
        # 计算权重 w0
        w0 = 1.0 - w1 - w2
        
        # 将权重堆叠并塑形
        # weights 形状: [B, H, W, 3] -> [B, 3, H, W] -> [B, 3, 1, H, W]
        weights = torch.stack([w0, w1, w2], dim=-1).permute(0, 3, 1, 2).unsqueeze(2)
        
    else:
        raise ValueError(f"未知的权重模式: {mode}。请选择 'barycentric_prox' 或 'barycentric'。")

    # --- 步骤 6: 加权求和 ---
    all_sampled_feats = torch.stack(sampled_features_list, dim=1) # 形状: [B, k, C, H, W]
    interpolated_feat = torch.sum(all_sampled_feats * weights, dim=1)

    return interpolated_feat


# ==============================================================================
# 3. Align 和 Align-Up 模块
# ==============================================================================

class AlignModule(nn.Module):
    def __init__(self, in_channels, k=3, align_corners=False, compress_ratio=4, use_scale=True):
        super().__init__()
        self.k = k
        self.align_corners = align_corners
        
        # 新增：用于压缩特征的1x1卷积
        compressed_channels = in_channels // compress_ratio
        self.compress_conv = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, 1, bias=True),
            # nn.BatchNorm2d(compressed_channels),
            # nn.ReLU(inplace=True) )
        )
        
        # offset_predictor现在接收压缩后的特征
        self.offset_predictor = nn.Conv2d(compressed_channels, k * 2, 3, padding=1)
        if use_scale:
            self.scale_predictor = nn.Conv2d(compressed_channels, k * 2, 3, padding=1)
            # nn.init.constant_(self.scale_predictor.weight, 0)
            # nn.init.constant_(self.scale_predictor.bias, 0)
        
        nn.init.constant_(self.offset_predictor.weight, 0)
        nn.init.constant_(self.offset_predictor.bias, 0)

    def forward(self, x, real_grid):
        B, C, H, W = x.shape
        
        # 核心改动：先压缩特征，再预测偏移量
        x_compressed = self.compress_conv(x)
        if hasattr(self, 'scale_predictor'):
            scale = self.scale_predictor(x_compressed).sigmoid_()
        else:
            scale = 1.0
        offsets = self.offset_predictor(x_compressed) * scale
        
        # 插值仍然使用原始的、全通道的特征x
        x_aligned = _perform_learnabel_non_uniform_approx_interpolation_with_real_grid(
            # x, offsets, real_grid, H, W, self.k, align_corners=self.align_corners, mode = 'barycentric_prox')
            x, offsets, real_grid, H, W, self.k, align_corners=self.align_corners, mode = 'barycentric')
        return x_aligned

class LPRMAlignUpModule(nn.Module):
    def __init__(self, low_res_channels, high_res_channels, kernel_size=3, align_corners=False, compress_ratio=4):
        super().__init__()
        self.align_corners = align_corners
        
        predictor_in_channels = low_res_channels + high_res_channels
        # compressed_channels = predictor_in_channels // compress_ratio
        compressed_channels = predictor_in_channels // compress_ratio

        self.compress_conv_low = nn.Sequential(
            nn.Conv2d(low_res_channels, compressed_channels, 1, bias=True),
            # nn.BatchNorm2d(compressed_channels),
            # nn.ReLU(inplace=True)
        )
        self.compress_conv_high = nn.Sequential(
            nn.Conv2d(high_res_channels, compressed_channels, 1, bias=True),
            # nn.BatchNorm2d(compressed_channels),
            # nn.ReLU(inplace=True)
        )
        
        # # offset_predictor接收压缩后的特征
        # self.offset_predictor = nn.Conv2d(compressed_channels, k * 2, 3, padding=1)

        # dilation = 1 is enough for FPN structure
        self.lprm = LPRM_efficient(channels=compressed_channels, kernel_size=kernel_size, upscale=2)
        self.lpr_conv = nn.Conv2d(compressed_channels, kernel_size ** 2, 3, padding=1)
        
        # nn.init.constant_(self.offset_predictor.weight, 0)
        # nn.init.constant_(self.offset_predictor.bias, 0)

    def forward(self, x_low, guidance_high_aligned):
        B, C, target_h, target_w = guidance_high_aligned.shape
        comporessed_low = self.compress_conv_low(x_low)
        comporessed_high = self.compress_conv_high(guidance_high_aligned)
        # local pixel relation
        lpr_feat = F.interpolate(self.lpr_conv(comporessed_low), scale_factor=2, mode='bilinear', align_corners=self.align_corners) \
            + F.interpolate(self.lpr_conv(comporessed_high), size=(x_low.size(-2) * 2, x_low.size(-1) * 2), mode='bilinear', align_corners=self.align_corners)
        x_low = self.lprm(lpr_feat, x_low)
        # print(x_low.shape)
        x_low_upsampled = F.interpolate(x_low, size=(target_h, target_w), mode='bilinear', align_corners=self.align_corners)
        
        return x_low_upsampled
    

    
import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保 mmcv 或 torchvision 中的算子可用
try:
    from mmcv.ops.carafe import carafe, CARAFE
except ImportError:
    print("警告: 无法从 mmcv.ops 导入 carafe。请确保 mmcv-full 已正确安装。")
    

class LPRM_efficient(nn.Module):
    """
    计算过程与/SFM_mask2former/mmseg_custom/models/segmentors/LPRM.py是相同的，但通过carafe算子节省现存
    使用 CARAFE 和 PixelShuffle 技巧模拟空洞邻域聚合的 LPRM。
    【已优化】：通过将 scale 维度折叠到 batch 维度来优化内存。
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, upscale=1, ):
        """
        初始化模块。

        Args:
            channels (int): 输入和输出 value_feat 的通道数。
            kernel_size (int): 邻域大小。
            dilation (int): 空洞率。
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.upscale = upscale
        
        # 模拟 dilation 的核心是 PixelShuffle 的 scale_factor
        self.scale_factor = dilation
        self.scale_sq = self.scale_factor * self.scale_factor
        
        # CARAFE 操作本身。我们让它在原地进行特征重组 (scale_factor=1)
        self.carafe_op = CARAFE(
            kernel_size=self.kernel_size,
            group_size=1,
            scale_factor=upscale
        )
        # self.pr_conv = nn.Conv2d(in_channels=channels, out_channels=kernel_size ** 2 * upscale ** 2, stride=1, dilation=self.dilation,
        #             kernel_size=kernel_size, bias=False, padding=dilation * (kernel_size // 2), groups=1)
        if self.scale_factor > 1:
            self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, mask_pred: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (已优化)。

        Args:
            mask_pred (torch.Tensor): 外部预测的重组权重 (mask)。
                                      形状: [B, k*k, H, W]。
            value_feat (torch.Tensor): 需要被提炼的值特征。
                                       形状: [B, C, H, W]。

        Returns:
            torch.Tensor: 提炼后的特征。形状: [B, C, H, W]。
        """
        # 如果 dilation 为 1，则无需模拟，直接应用 CARAFE
        # mask_pred = self.pr_conv(mask_pred)
        # print(mask_pred.shape)
        # print(value_feat.shape)
        if self.scale_factor == 1:
            return self.carafe_op(value_feat, mask_pred.softmax(dim=1))

        # --- 模拟空洞操作 ---
        B, C, H, W = value_feat.shape
        k_sq = self.kernel_size * self.kernel_size
        s = self.scale_factor
        Hs, Ws = H // s, W // s # Unshuffle 后的空间尺寸
        
        # 1. Unshuffle 并折叠到 Batch 维度
        # [B, C, H, W] -> [B, C * s^2, Hs, Ws]
        unshuffled_value = self.pixel_unshuffle(value_feat)
        # [B, C*s^2, Hs, Ws] -> [B, C, s^2, Hs, Ws] -> [B, s^2, C, Hs, Ws] -> [B*s^2, C, Hs, Ws]
        unshuffled_value = unshuffled_value.view(B, C, self.scale_sq, Hs, Ws).permute(0, 2, 1, 3, 4).reshape(B * self.scale_sq, C, Hs, Ws)

        # [B, k*k, H, W] -> [B, k*k * s^2, Hs, Ws]
        unshuffled_mask = self.pixel_unshuffle(mask_pred)
        # [B, k*k*s^2, Hs, Ws] -> [B, k*k, s^2, Hs, Ws] -> [B, s^2, k*k, Hs, Ws] -> [B*s^2, k*k, Hs, Ws]
        unshuffled_mask = unshuffled_mask.view(B, k_sq, self.scale_sq, Hs, Ws).permute(0, 2, 1, 3, 4).reshape(B * self.scale_sq, k_sq, Hs, Ws)
        
        # 2. 在折叠后的 Batch 上应用 CARAFE (更高效)
        # 输入和输出的 H, W 都是 Hs, Ws
        refined_unshuffled = self.carafe_op(unshuffled_value, unshuffled_mask.softmax(dim=1).view(n, -1, h, w))
        
        # 3. 逆操作：从 Batch 维度恢复并 PixelShuffle 回原始分辨率
        # [B*s^2, C, Hs, Ws] -> [B, s^2, C, Hs, Ws]
        refined_unshuffled = refined_unshuffled.view(B, self.scale_sq, C, Hs, Ws)
        # [B, s^2, C, Hs, Ws] -> [B, C, s^2, Hs, Ws] -> [B, C*s^2, Hs, Ws]
        refined_shuffled = refined_unshuffled.permute(0, 2, 1, 3, 4).reshape(B, C * self.scale_sq, Hs, Ws)
        
        # [B, C*s^2, Hs, Ws] -> [B, C, H, W]
        refined_feat = self.pixel_shuffle(refined_shuffled)
        
        return refined_feat


class MSLPRM_efficient(nn.Module):
    """
    使用 CARAFE 实现的多尺度局部像素关系模块。
    """
    def __init__(self, in_channels: int, refine_channels: int = 16, dilations: list = [1], upscale=1, kernel_size: int = 3):
        """
        初始化 MSLPRM_efficient。
        """
        super().__init__()
        
        mask_channels = 1 * kernel_size * kernel_size
        if upscale > 1: assert len(dilations) == 1
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(kernel_size * kernel_size),
            # nn.ReLU(inplace=True),
        )
        self.lpr_modules = nn.ModuleList([
            LPRM_efficient(
                channels=kernel_size ** 2,
                kernel_size=kernel_size,
                dilation=d,
                upscale=upscale,
            ) for d in dilations
        ])

        self.lpr_convs = nn.ModuleList([
            nn.Conv2d(in_channels=kernel_size ** 2 * upscale ** 2 if i == 1 else kernel_size ** 2, 
                      out_channels=kernel_size ** 2 * upscale ** 2 if i == 0 else kernel_size ** 2, 
                      kernel_size=kernel_size, bias=False, padding=dilation * (kernel_size // 2), groups=1,
                      stride=1, dilation=dilations[i],
            ) for i in range(len(dilations))
        ])

    def forward(self, x: torch.Tensor, prediction_map: torch.Tensor, residual: bool = False) -> torch.Tensor:
        mask_pred_logits = self.compress(x)
        
        refined_map = prediction_map
        for (lpr_conv, lprm) in (self.lpr_convs, self.lpr_modules):
            # res = lprm(mask_pred=mask_pred_logits, value_feat=refined_map)
            # print(len(res))
            mask_pred_logits = lpr_conv(mask_pred_logits)
            refined_map = lprm(mask_pred=mask_pred_logits, value_feat=refined_map)
        if residual:
            return prediction_map + refined_map
        else: return refined_map

# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == '__main__':
    # 确保环境中 mmcv 安装正确
    try:
        from mmcv.ops import carafe, CARAFE
        print("mmcv.ops.carafe 导入成功!")
    except ImportError:
        print("错误: 无法导入 carafe。请确保 mmcv-full 已正确安装。")
        exit()

    # --- 模拟输入 ---
    BATCH_SIZE = 4
    IN_CHANNELS = 256
    NUM_CLASSES = 19
    H, W = 64, 64

    decoder_feature = torch.randn(BATCH_SIZE, IN_CHANNELS, H, W).cuda()
    initial_prediction = torch.randn(BATCH_SIZE, NUM_CLASSES, H, W).cuda()

    # --- 初始化 MSLPRM_efficient 模块 ---
    dilations = [1, 2, 4, 8] # 定义多尺度的空洞率
    kernel_size = 5 # CARAFE 通常使用稍大的 kernel
    
    msLPRM_efficient_module = MSLPRM_efficient(
        in_channels=IN_CHANNELS,
        refine_channels=NUM_CLASSES,
        dilations=dilations,
        kernel_size=kernel_size
    ).cuda()

    # --- 执行前向传播 ---
    print(f"\n输入特征形状: {decoder_feature.shape}")
    print(f"输入预测图形状: {initial_prediction.shape}")
    
    refined_prediction_carafe = msLPRM_efficient_module(decoder_feature, initial_prediction)
    
    print("-" * 30)
    print(f"CARAFE实现的LPRM输出形状: {refined_prediction_carafe.shape}")
    
    assert refined_prediction_carafe.shape == initial_prediction.shape
    print("CARAFE (Batch-Folding优化) 版本代码运行成功，输出形状正确！")
    
    num_params_carafe = sum(p.numel() for p in msLPRM_efficient_module.parameters() if p.requires_grad)
    print(f"MSLPRM_efficient 模块参数量: {num_params_carafe / 1e6:.2f} M")