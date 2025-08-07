import torch
import numpy as np
from PIL import Image
import cv2

class PatternFillNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像（作为填充区域的参考）
                "pattern": ("IMAGE",),  # 图案图像
                "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),  # 角度 -180到180
                "scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0}),  # 缩放 1%到1000%
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pattern_fill"
    CATEGORY = "ZVNodes/photoshop"

    def tensor_to_numpy(self, tensor):
        """将 Tensor 转换为 numpy array"""
        image_np = tensor.squeeze().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return image_np

    def numpy_to_tensor(self, image_np):
        """将 numpy array 转换为 Tensor"""
        image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor

    def resize_pattern(self, pattern, scale):
        """根据缩放比例调整图案大小"""
        height, width = pattern.shape[:2]
        new_width = max(1, int(width * scale / 100))
        new_height = max(1, int(height * scale / 100))
        
        if new_width > 0 and new_height > 0:
            resized_pattern = cv2.resize(pattern, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_pattern
        else:
            return pattern

    def rotate_pattern(self, pattern, angle):
        """旋转图案"""
        if abs(angle) < 0.1:
            return pattern
            
        height, width = pattern.shape[:2]
        center = (width // 2, height // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新图像的边界
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # 调整旋转中心
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # 执行旋转
        rotated_pattern = cv2.warpAffine(pattern, rotation_matrix, (new_width, new_height), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rotated_pattern

    def tile_pattern(self, pattern, target_width, target_height):
        """平铺图案以填充目标区域"""
        if pattern is None or pattern.size == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
        pattern_height, pattern_width = pattern.shape[:2]
        
        if pattern_width == 0 or pattern_height == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算需要的重复次数
        repeat_x = max(1, int(np.ceil(target_width / pattern_width)) + 1)
        repeat_y = max(1, int(np.ceil(target_height / pattern_height)) + 1)
        
        # 创建大图案
        if len(pattern.shape) == 3:
            tiled_pattern = np.tile(pattern, (repeat_y, repeat_x, 1))
        else:
            tiled_pattern = np.tile(pattern, (repeat_y, repeat_x))
            # 如果是灰度图，转换为RGB
            if len(tiled_pattern.shape) == 2:
                tiled_pattern = cv2.cvtColor(tiled_pattern, cv2.COLOR_GRAY2RGB)
        
        # 裁剪到目标大小
        tiled_pattern = tiled_pattern[:target_height, :target_width]
        
        return tiled_pattern

    def apply_pattern_fill(self, image, pattern, angle, scale):
        # 转换图像格式
        base_image = self.tensor_to_numpy(image)
        pattern_image = self.tensor_to_numpy(pattern)
        
        # 获取目标尺寸
        target_height, target_width = base_image.shape[:2]
        
        # 应用缩放
        if abs(scale - 100.0) > 0.1:
            pattern_image = self.resize_pattern(pattern_image, scale)
        
        # 应用旋转
        if abs(angle) > 0.1:
            pattern_image = self.rotate_pattern(pattern_image, angle)
        
        # 平铺图案
        tiled_pattern = self.tile_pattern(pattern_image, target_width, target_height)
        
        # 确保输出是RGB格式
        if len(tiled_pattern.shape) == 2:
            tiled_pattern = cv2.cvtColor(tiled_pattern, cv2.COLOR_GRAY2RGB)
        
        # 转换回 Tensor
        result_tensor = self.numpy_to_tensor(tiled_pattern)
        
        return (result_tensor,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "PatternFillNodeZV": PatternFillNodeZV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatternFillNodeZV": "Photoshop Pattern Fill"
}