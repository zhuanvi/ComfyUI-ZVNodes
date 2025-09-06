

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import comfy


class ProductionDisplacementMapNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "displacement_map": ("IMAGE",),
                "horizontal_scale": ("FLOAT", {"default": 10.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "vertical_scale": ("FLOAT", {"default": 10.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "displacement_type": (["stretch", "repeat", "reflect"], {"default": "stretch"}),
                "channel_mode": (["luminance", "red_green"], {"default": "luminance"}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_displacement"
    CATEGORY = "ZVNodes/filter"

    def apply_displacement(self, image, displacement_map, horizontal_scale, vertical_scale, 
                          displacement_type, channel_mode, mask=None, mask_strength=1.0):
        # 调整位移图尺寸
        if displacement_map.shape[1:3] != image.shape[1:3]:
            displacement_map = comfy.utils.common_upscale(
                displacement_map.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center"
            ).movedim(1,-1)
        
        # 处理遮罩
        if mask is not None:
            # 调整遮罩尺寸
            if mask.shape[1:3] != image.shape[1:3]:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), 
                    size=(image.shape[1], image.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # 应用遮罩强度
            if mask_strength < 1.0:
                mask = mask * mask_strength
        
        # 处理批量图像
        result = torch.zeros_like(image)
        batch_size = image.shape[0]
        disp_batch_size = displacement_map.shape[0]
        mask_batch_size = mask.shape[0] if mask is not None else 0
        
        for i in range(batch_size):
            disp_idx = i if i < disp_batch_size else 0
            mask_idx = i if mask is not None and i < mask_batch_size else 0
            
            result[i] = self.displace_single_image(
                image[i], 
                displacement_map[disp_idx], 
                horizontal_scale, 
                vertical_scale, 
                displacement_type,
                channel_mode,
                mask[mask_idx] if mask is not None else None
            )
        
        return (result,)

    def displace_single_image(self, image_tensor, disp_tensor, h_scale, v_scale, disp_type, channel_mode, mask=None):
        device = image_tensor.device
        h, w, c = image_tensor.shape
        
        # 根据通道模式计算位移
        if channel_mode == "red_green" and disp_tensor.shape[2] >= 2:
            # 使用红色通道控制水平位移，绿色通道控制垂直位移
            disp_x = disp_tensor[:, :, 0]  # 红色通道
            disp_y = disp_tensor[:, :, 1]  # 绿色通道
        else:
            # 使用亮度控制两个方向的位移
            if disp_tensor.shape[2] == 3:
                disp_gray = 0.2989 * disp_tensor[:, :, 0] + 0.5870 * disp_tensor[:, :, 1] + 0.1140 * disp_tensor[:, :, 2]
            else:
                disp_gray = disp_tensor[:, :, 0]
            disp_x = disp_gray
            disp_y = disp_gray
        
        # 将位移值从[0,1]映射到[-1,1]
        disp_x = (disp_x - 0.5) * 2
        disp_y = (disp_y - 0.5) * 2
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 计算新坐标
        new_x = x_coords + disp_x * h_scale
        new_y = y_coords + disp_y * v_scale
        
        # 处理边界
        if disp_type == "stretch":
            new_x = torch.clamp(new_x, 0, w-1)
            new_y = torch.clamp(new_y, 0, h-1)
        elif disp_type == "repeat":
            new_x = new_x % w
            new_y = new_y % h
        elif disp_type == "reflect":
            new_x = torch.abs((new_x % (2*w)) - w)
            new_y = torch.abs((new_y % (2*h)) - h)
            new_x = torch.clamp(new_x, 0, w-1)
            new_y = torch.clamp(new_y, 0, h-1)
        
        # 准备网格采样
        grid_x = (new_x / (w-1)) * 2 - 1
        grid_y = (new_y / (h-1)) * 2 - 1
        
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        
        # 调整图像张量形状为NCHW
        image_nchw = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # 使用网格采样进行插值
        displaced_image = torch.nn.functional.grid_sample(
            image_nchw, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # 调整回原始形状
        displaced_image = displaced_image.squeeze(0).permute(1, 2, 0)
        
        # 应用遮罩
        if mask is not None:
            # 确保遮罩形状正确
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(2)
            
            # 扩展遮罩维度以匹配图像通道数
            if mask.shape[2] == 1:
                mask_expanded = mask.expand(-1, -1, c)
            else:
                mask_expanded = mask
            
            # 混合原始图像和位移后的图像
            result = image_tensor * (1 - mask_expanded) + displaced_image * mask_expanded
        else:
            result = displaced_image
        
        return result

