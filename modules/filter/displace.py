import torch
import numpy as np
from torchvision.transforms import functional as TF

# comfyui_displace/displacement_node.py

import torch
import numpy as np
from torchvision import transforms
from torch import nn

class ProductionDisplacementMapNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),                    # (B, H, W, 3)
                "displacement_map": ("IMAGE",),         # (B, H_d, W_d, C)
                "horizontal_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 0.5
                }),
                "vertical_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 0.5
                }),
                "fit_method": (["stretch", "tile"], {"default": "stretch"}),
                "undefined_area": (["replicate", "wrap"], {"default": "replicate"}),
                "edge_smoothing": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Smooth displacement map to reduce tearing"
                }),
                "max_displacement_limit": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Max displacement in pixels. 0 = no limit"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to limit displacement effect area"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_displacement"
    CATEGORY = "image/postprocessing"
    DESCRIPTION = "High-quality displacement with edge padding to prevent tearing."

    def apply_displacement(self, image, displacement_map, horizontal_scale, vertical_scale,
                         fit_method, undefined_area, edge_smoothing, max_displacement_limit,
                         mask=None):

        device = image.device
        batch_size, height, width, channels = image.shape

        # 处理 displacement_map batch
        if displacement_map.shape[0] == 1:
            disp = displacement_map.repeat(batch_size, 1, 1, 1)
        else:
            disp = displacement_map

        # 调整尺寸
        disp = disp.permute(0, 3, 1, 2)  # -> (B, C, H_d, W_d)

        if fit_method == "stretch":
            disp_resized = transforms.Resize((height, width), 
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           antialias=True)(disp)
        else:  # tile
            tiles_x = (width // disp.shape[3]) + 2
            tiles_y = (height // disp.shape[2]) + 2
            disp_tiled = disp.tile(1, 1, tiles_y, tiles_x)
            disp_resized = transforms.CenterCrop((height, width))(disp_tiled)

        disp_resized = disp_resized.permute(0, 2, 3, 1)  # (B, H, W, C)

        # 提取 R/G 通道
        if disp_resized.shape[-1] >= 2:
            disp_x = disp_resized[..., 0]  # Red
            disp_y = disp_resized[..., 1]  # Green
        else:
            gray = disp_resized[..., 0]
            disp_x = gray
            disp_y = gray

        # 归一化到 [-1, 1]（中性灰 = 0.5）
        disp_x = (disp_x - 0.5) * 2.0
        disp_y = (disp_y - 0.5) * 2.0

        # 应用缩放
        disp_x = disp_x * horizontal_scale
        disp_y = disp_y * vertical_scale

        # 【优化1】限制最大位移
        if max_displacement_limit > 0:
            disp_x = torch.clamp(disp_x, -max_displacement_limit, max_displacement_limit)
            disp_y = torch.clamp(disp_y, -max_displacement_limit, max_displacement_limit)

        # 【优化2】平滑置换图（防撕裂）
        if edge_smoothing > 0:
            kernel_size = int(2 * round(edge_smoothing) + 1)
            kernel_size = max(3, kernel_size)  # 最小 3x3
            blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=edge_smoothing)
            disp_x = blur(disp_x.unsqueeze(1)).squeeze(1)
            disp_y = blur(disp_y.unsqueeze(1)).squeeze(1)

        # 创建基础网格
        grid_x, grid_y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing='xy'
        )
        grid_x = grid_x.float().unsqueeze(0).expand(batch_size, -1, -1)
        grid_y = grid_y.float().unsqueeze(0).expand(batch_size, -1, -1)

        # 计算目标坐标
        target_x = grid_x + disp_x
        target_y = grid_y + disp_y

        # 【核心优化】边缘扩展策略
        pad = max(32, int(max(horizontal_scale, vertical_scale) * 1.2))  # 动态 padding
        pad = min(pad, min(height, width) // 2)  # 不超过图像一半

        # 扩展图像
        img_tensor = image.permute(0, 3, 1, 2)  # (B, 3, H, W)
        img_padded = nn.functional.pad(img_tensor, (pad, pad, pad, pad), mode='replicate')

        # 调整 flow 坐标到 padded 空间
        flow_x_padded = 2 * (target_x + pad) / (width + 2*pad - 1) - 1
        flow_y_padded = 2 * (target_y + pad) / (height + 2*pad - 1) - 1
        flow_padded = torch.stack([flow_x_padded, flow_y_padded], dim=-1)  # (B, H, W, 2)

        # 【处理 wrap 模式】
        if undefined_area == "wrap":
            # 使用模运算实现循环
            x_unnorm = (flow_x_padded + 1) * 0.5 * (width + 2*pad)
            y_unnorm = (flow_y_padded + 1) * 0.5 * (height + 2*pad)

            x_wrap = (x_unnorm % (width + 2*pad)) / (width + 2*pad) * 2 - 1
            y_wrap = (y_unnorm % (height + 2*pad)) / (height + 2*pad) * 2 - 1

            flow_padded = torch.stack([x_wrap, y_wrap], dim=-1)

        # 采样
        result_padded = nn.functional.grid_sample(
            img_padded, flow_padded,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # (B, 3, H+2pad, W+2pad)

        # 裁剪回原始尺寸
        result = result_padded[:, :, pad:pad+height, pad:pad+width]

        # 【可选】应用 mask
        if mask is not None and mask.shape[0] == batch_size:
            mask_exp = mask.unsqueeze(1)  # (B, 1, H, W)
            orig = img_tensor
            result = mask_exp * result + (1 - mask_exp) * orig

        # 转回 (B, H, W, 3)
        result = result.permute(0, 2, 3, 1)
        result = torch.clamp(result, 0.0, 1.0)
        result = torch.nan_to_num(result, 0.0)

        return (result,)




class DesaturateNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),                    # 输入图像 (B, H, W, 3)
                "output_channels": (["1", "3"], {"default": "1"}),  # 输出通道数
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "desaturate"
    CATEGORY = "ZVNodes/filter"
    DESCRIPTION = "Convert image to grayscale (desaturate), useful for generating displacement maps."

    def desaturate(self, image, output_channels):
        """
        将图像去色（转为灰度图）
        :param image: (B, H, W, 3) in [0,1]
        :param output_channels: "1" 或 "3"
        :return: (B, H, W, C) 灰度图，值范围 [0,1]
        """
        # RGB 权重（ITU-R BT.601）
        # 也可用 BT.709: [0.2126, 0.7152, 0.0722]
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)  # (3,)

        # 转为 (B, H, W, 3) -> (B, H, W)
        gray = torch.tensordot(image, weights, dims=1)  # (B, H, W)

        # 扩展为 (B, H, W, 1)
        gray = gray.unsqueeze(-1)  # (B, H, W, 1)

        if output_channels == "1":
            # 单通道输出
            return (gray,)
        else:
            # 三通道复制（模拟灰度图）
            gray_3 = gray.repeat(1, 1, 1, 3)  # (B, H, W, 3)
            return (gray_3,)


# 可选：增强版 —— 直接生成标准化置换图（用于位移）
class GrayToDisplacementMapNodeZV:
    """
    将灰度图标准化为 [0,1] 范围的置换图
    可用于 Displacement 节点的 displacement_map 输入
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": False}),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "ZVNodes/filter"
    DESCRIPTION = "Convert any image to a normalized displacement map with brightness/contrast control."

    def process(self, image, invert, contrast, brightness):
        # 去色
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)
        disp = torch.tensordot(image, weights, dims=1).unsqueeze(-1)  # (B, H, W, 1)

        # 调整对比度和亮度
        disp = disp * contrast + brightness
        disp = torch.clamp(disp, 0.0, 1.0)

        # 反相
        if invert:
            disp = 1.0 - disp

        # 扩展为 3 通道（ComfyUI 多数图像为 3 通道）
        disp_3 = disp.repeat(1, 1, 1, 3)

        return (disp_3,)

