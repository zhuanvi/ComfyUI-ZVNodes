

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


class ProductionDisplacementMapNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "displacement_map": ("IMAGE",),
                "horizontal_scale": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 500.0, "step": 0.5
                }),
                "vertical_scale": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 500.0, "step": 0.5
                }),
                "fit_method": (["stretch", "tile"], {"default": "stretch"}),
                "undefined_area": (["replicate", "wrap"], {"default": "replicate"}),
                "edge_smoothing": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1
                }),
                "max_displacement_limit": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 1.0
                }),
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_displacement"
    CATEGORY = "ZVNodes/filter"
    DESCRIPTION = "High-quality displacement with correct output size."

    def preprocess_mask(self, mask, batch_size, height, width):
        device = mask.device
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        elif mask.dim() == 3:
            if mask.shape[-1] == 4:
                mask = mask[..., 3:4]  # 取 alpha
            else:
                mask = mask.unsqueeze(-1)  # (B, H, W, 1)
        elif mask.dim() == 4:
            if mask.shape[-1] == 4:
                mask = mask[..., 3:4]  # RGBA -> A
            elif mask.shape[-1] == 3:
                # RGB 转灰度
                weights = torch.tensor([0.299, 0.587, 0.114], device=device)
                mask = torch.tensordot(mask, weights, dims=1).unsqueeze(-1)
            else:
                mask = mask  # 假设是 (B, H, W, 1)

        # 调整大小（可选）
        if mask.shape[1] != height or mask.shape[2] != width:
            mask = F.interpolate(mask.permute(0,3,1,2), size=(height, width), mode='bilinear')
            mask = mask.permute(0,2,3,1)

        # 扩展 batch
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.repeat(batch_size, 1, 1, 1)

        return torch.clamp(mask, 0.0, 1.0)

    def apply_displacement(self, image, displacement_map, horizontal_scale, vertical_scale,
                         fit_method, undefined_area, edge_smoothing, max_displacement_limit,
                         mask=None):

        device = image.device
        batch_size, height, width, channels = image.shape

        # 处理 displacement_map
        if displacement_map.shape[0] == 1:
            disp = displacement_map.repeat(batch_size, 1, 1, 1)
        else:
            disp = displacement_map

        disp = disp.permute(0, 3, 1, 2)  # (B, C, H_d, W_d)

        if fit_method == "stretch":
            disp_resized = transforms.Resize(
                (height, width),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )(disp)
        else:  # tile
            tiles_x = (width // disp.shape[3]) + 2
            tiles_y = (height // disp.shape[2]) + 2
            disp_tiled = disp.tile(1, 1, tiles_y, tiles_x)
            disp_resized = transforms.CenterCrop((height, width))(disp_tiled)

        disp_resized = disp_resized.permute(0, 2, 3, 1)  # (B, H, W, C)

        # 提取位移通道
        if disp_resized.shape[-1] >= 2:
            disp_x = disp_resized[..., 0]
            disp_y = disp_resized[..., 1]
        else:
            gray = disp_resized[..., 0]
            disp_x = gray
            disp_y = gray

        # 归一化到 [-1,1]
        disp_x = (disp_x - 0.5) * 2.0
        disp_y = (disp_y - 0.5) * 2.0

        # 应用缩放
        disp_x = disp_x * horizontal_scale
        disp_y = disp_y * vertical_scale

        # 限制最大位移
        if max_displacement_limit > 0:
            disp_x = torch.clamp(disp_x, -max_displacement_limit, max_displacement_limit)
            disp_y = torch.clamp(disp_y, -max_displacement_limit, max_displacement_limit)

        # 平滑
        if edge_smoothing > 0:
            k = int(2 * round(edge_smoothing) + 1)
            k = max(3, min(31, k))  # 限制 kernel 大小
            blur = transforms.GaussianBlur(kernel_size=k, sigma=edge_smoothing)
            disp_x = blur(disp_x.unsqueeze(1)).squeeze(1)
            disp_y = blur(disp_y.unsqueeze(1)).squeeze(1)

        # 创建网格（像素坐标）
        grid_x = torch.linspace(0, width - 1, width, device=device)
        grid_y = torch.linspace(0, height - 1, height, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')  # (H, W)
        grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)
        grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)

        # 目标坐标
        target_x = grid_x + disp_x  # (B, H, W)
        target_y = grid_y + disp_y

        # 扩展 padding
        pad = max(32, int(max(horizontal_scale, vertical_scale) * 1.2))
        pad = min(pad, min(height, width) // 2)

        img_tensor = image.permute(0, 3, 1, 2)  # (B, 3, H, W)
        img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode='replicate')  # (B, 3, H+2p, W+2p)

        # ✅ 正确归一化到 [-1,1]（align_corners=False）
        # padded 图像的新宽高
        padded_w = width + 2 * pad
        padded_h = height + 2 * pad

        # 将目标坐标转换为 padded 图像上的 [-1,1] 坐标
        norm_x = 2 * (target_x + pad) / (padded_w - 1) - 1  # 注意：使用 (size-1)
        norm_y = 2 * (target_y + pad) / (padded_h - 1) - 1

        # 处理 wrap 模式
        if undefined_area == "wrap":
            # 转回像素坐标，取模，再归一化
            px_x = (norm_x + 1) * 0.5 * padded_w
            px_y = (norm_y + 1) * 0.5 * padded_h
            px_x_wrap = torch.fmod(px_x, padded_w)
            px_y_wrap = torch.fmod(px_y, padded_h)
            norm_x = 2 * px_x_wrap / padded_w - 1
            norm_y = 2 * px_y_wrap / padded_h - 1

        flow = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)

        # 采样
        result_padded = F.grid_sample(
            img_padded, flow,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False  # ✅ 保持一致
        )  # (B, 3, H, W) —— 输出尺寸自动为 (B,3,H,W)

        # ✅ 不需要 crop！因为 flow 指向 padded 区域，但输出仍是 (H, W)
        result = result_padded  # 尺寸已是 (B, 3, H, W)

        # 转回 (B, H, W, 3)
        result = result.permute(0, 2, 3, 1)
        result = torch.clamp(result, 0.0, 1.0)
        result = torch.nan_to_num(result, 0.0)

        # 可选 mask
        if mask is not None:
            mask = self.preprocess_mask(mask, batch_size, height, width)
            result = mask * result + (1 - mask) * image

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

