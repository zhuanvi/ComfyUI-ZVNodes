from .utils import generate_node_mappings

import random
import numpy as np
from PIL import Image, ImageDraw
import torch

class TriangleCharacterLayoutZV:
    """
    ComfyUI 自定义节点：在不规则三角区域内自动生成不重叠的人物坐标布局
    - 支持一个主体人物（用于动作控制，位于三角中心）
    - 其余为自动生成的配角，随机布局但不重叠
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas_width": ("INT", {"default": 1024, "min": 512, "max": 4096}),
                "canvas_height": ("INT", {"default": 800, "min": 512, "max": 4096}),
                "triangle_json": ("JSON", ),
                "center_box": ("VEC2", {"default": [160, 280]}),
                "other_box": ("VEC2", {"default": [128, 256]}),
                "jitter_enabled": ("BOOLEAN", {"default": False}),
                "total_characters": ("INT", {"default": 5, "min": 1, "max": 20}),
                "preview": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "jitter_range": ("VEC2", {"default": [20, 20]})
            }
        }

    RETURN_TYPES = ("JSON", "IMAGE")
    RETURN_NAMES = ("layout_json", "preview")
    FUNCTION = "generate"
    CATEGORY = "ZVNodes/Layout"

    def generate(self, canvas_width, canvas_height,
                 triangle_json,
                 jitter_enabled, jitter_range,
                 center_box, other_box, total_characters, preview):

        def jitter(point, jitter_range):
            jitter_x = int(jitter_range[0])
            jitter_y = int(jitter_range[1])
            return (
                point[0] + random.randint(-jitter_x, jitter_x),
                point[1] + random.randint(-jitter_y, jitter_y)
            )

        def is_inside_triangle(pt, a, b, c):
            x, y = pt
            def area(x1,y1,x2,y2,x3,y3):
                return abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2.0)
            A = area(*a, *b, *c)
            A1 = area(x, y, *b, *c)
            A2 = area(*a, x, y, *c)
            A3 = area(*a, *b, x, y)
            return abs(A - (A1 + A2 + A3)) < 1e-1

        def overlaps(pt, others, size):
            x, y = pt
            w, h = size
            for ox, oy, ow, oh in others:
                if abs(x - ox) < (w + ow) // 2 and abs(y - oy) < (h + oh) // 2:
                    return True
            return False

        if not isinstance(triangle_json,list) or  len(triangle_json)<3:
            raise ValueError(f"triangle_json格式错误")
    
        triangle_point_A = [ triangle_json[0]["x"],triangle_json[0]["y"] ]
        triangle_point_B = [ triangle_json[1]["x"],triangle_json[1]["y"] ]
        triangle_point_C = [ triangle_json[2]["x"],triangle_json[2]["y"] ]

        # 应用抖动（如果启用）
        A = jitter(triangle_point_A, jitter_range) if jitter_enabled else triangle_point_A
        B = jitter(triangle_point_B, jitter_range) if jitter_enabled else triangle_point_B
        C = jitter(triangle_point_C, jitter_range) if jitter_enabled else triangle_point_C

        # 主体人物放三角形中心
        centroid = (
            int((A[0] + B[0] + C[0]) / 3),
            int((A[1] + B[1] + C[1]) / 3)
        )
        results = [(centroid[0], centroid[1], center_box[0], center_box[1])]

        # 随机放置其余人物
        attempts = 0
        while len(results) < total_characters and attempts < 1000:
            x = random.randint(0, canvas_width)
            y = random.randint(0, canvas_height)
            if not is_inside_triangle((x, y), A, B, C):
                attempts += 1
                continue
            if overlaps((x, y), results, other_box):
                attempts += 1
                continue
            results.append((x, y, other_box[0], other_box[1]))
            attempts += 1

        layout = {
            "positions": [
                {"x": x, "y": y, "width": w, "height": h} for x, y, w, h in results
            ],
            "primary_pose_index": 0
        }

        # 生成预览图
        img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(A), tuple(B), tuple(C)], outline=(0, 255, 0, 255), width=2)
        for i, (x, y, w, h) in enumerate(results):
            color = (255, 0, 0, 255) if i == 0 else (255, 255, 255, 255)
            draw.rectangle([(x - w//2, y - h//2), (x + w//2, y + h//2)], outline=color, width=2)
            draw.text((x - 6, y - 10), str(i+1), fill=color)

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        return (layout, img)

class JsonListToMaskZV:
    """将JSON列表中的矩形渲染为遮罩图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list": ("JSON",),
                "width": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 8192,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 8192,
                    "step": 1
                }),
                "invert": (["false", "true"], {
                    "default": "false"
                })
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "render_mask"
    CATEGORY = "ZVNodes/layout"
    DESCRIPTION = "将JSON列表中的矩形渲染为遮罩图像"

    def render_mask(self, json_list, width, height, invert):
        # 验证输入类型
        if not isinstance(json_list, list):
            raise ValueError("输入必须是JSON列表")
        
        # 创建全黑图像
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        
        # 遍历所有矩形并绘制
        for rect in json_list:
            try:
                # 提取矩形参数
                w = rect.get('width', 0)
                h = rect.get('height', 0)
                x = int(rect.get('x', 0) - w / 2)
                y = int(rect.get('y', 0) - h / 2)
                
                
                # 确保坐标在有效范围内
                x = max(0, min(x, width))
                y = max(0, min(y, height))
                w = max(0, min(w, width - x))
                h = max(0, min(h, height - y))
                
                # 绘制白色矩形
                draw.rectangle([(x, y), (x + w, y + h)], fill=255)
            except Exception as e:
                raise ValueError(f"无效矩形数据: {rect} - {str(e)}")
        
        # 转换为numpy数组
        mask_array = np.array(mask_image).astype(np.float32) / 255.0
        
        # 如果需要反转遮罩
        if invert == "true":
            mask_array = 1.0 - mask_array
        
        # 转换为PyTorch张量
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        
        # 创建RGB图像用于预览
        rgb_image = Image.new("RGB", (width, height))
        rgb_draw = ImageDraw.Draw(rgb_image)
        for rect in json_list:
            x = rect.get('x', 0)
            y = rect.get('y', 0)
            w = rect.get('width', 0)
            h = rect.get('height', 0)
            rgb_draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255))
        
        # 转换为ComfyUI图像格式
        rgb_array = np.array(rgb_image).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_array)[None,]
        
        return (mask_tensor, rgb_tensor)

NODE_CONFIG = {
    "TriangleCharacterLayoutZV": {"class": TriangleCharacterLayoutZV, "name": "Triangle Character Layout"},
    "JsonListToMaskZV": {"class": JsonListToMaskZV, "name": "Json List To Mask"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)