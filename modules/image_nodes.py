import os
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import torch
import torchvision.transforms.functional as F
import numpy as np
import torchvision.transforms as transforms
import json
import ast
import requests
import urllib.parse
from .utils import generate_node_mappings, pil2tensor
from pillow_heif import register_heif_opener
register_heif_opener()


class UniversalBBOXToMaskZV:
    """
    通用BBOX转Mask节点
    支持XYXY、XYWH、CXCYWH格式，归一化和非归一化坐标
    忽略labels项，自动检测格式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_string": ("STRING", {
                    "multiline": True,
                    "default": "[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.2]]"
                }),
                "image_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "image_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "mask_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "format": (["auto", "xyxy", "xywh", "cxcywh"], {
                    "default": "auto",
                    "tooltip": "xyxy: [x1, y1, x2, y2]\nxywh: [x, y, width, height]\ncxcywh: [center_x, center_y, width, height]"
                }),
                "normalized": (["auto", "yes", "no"], {
                    "default": "auto"
                }),
                "fill_mode": (["rectangle", "ellipse", "gaussian", "soft_rectangle"], {
                    "default": "rectangle"
                }),
                "border_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "边界宽度，0表示填充整个区域"
                }),
                "min_bbox_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "最小bbox尺寸（像素），小于此值的bbox将被忽略"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "base_mask": ("MASK",),
                "debug_mode": (["no", "yes"], {
                    "default": "no"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "LIST", "LIST", "STRING")
    RETURN_NAMES = ("mask", "visualization", "info", "pixel_bbox_list", "normalized_bbox_list", "formatted_string")
    FUNCTION = "convert_bbox_to_mask"
    CATEGORY = "mask/bbox"
    
    def convert_bbox_to_mask(self, bbox_string, image_width, image_height, mask_value, 
                            format, normalized, fill_mode, border_width, min_bbox_size,
                            reference_image=None, base_mask=None, debug_mode="no"):
        try:
            if debug_mode == "yes":
                print(f"=== UniversalBBOXToMask Debug ===\n")
                print(f"Input bbox_string: {bbox_string}")
                print(f"Image size: {image_width}x{image_height}")
                print(f"Format: {format}, Normalized: {normalized}")
            
            # 解析bbox字符串，提取boxes（忽略labels）
            boxes = self.parse_bbox_string(bbox_string)
            
            if debug_mode == "yes":
                print(f"Parsed boxes: {boxes}")
            
            # 获取图像尺寸
            if reference_image is not None:
                batch_size, img_height, img_width, _ = reference_image.shape
                image_height = img_height
                image_width = img_width
            elif base_mask is not None:
                batch_size, img_height, img_width = base_mask.shape
                image_height = img_height
                image_width = img_width
            
            # 创建或获取基础mask
            if base_mask is not None:
                mask = base_mask.clone()
            else:
                mask = torch.zeros((1, image_height, image_width), dtype=torch.float32)
            
            # 检测格式和归一化状态
            detected_format, detected_normalized = self.detect_format_and_normalization(
                boxes, image_width, image_height
            )
            
            # 使用用户指定的格式或自动检测的格式
            if format == "auto":
                format = detected_format
            if normalized == "auto":
                normalized = detected_normalized
            
            if debug_mode == "yes":
                print(f"Detected format: {detected_format}, Detected normalized: {detected_normalized}")
                print(f"Final format: {format}, Final normalized: {normalized}")
            
            # 处理所有bbox
            processed_boxes = self.process_boxes(
                boxes, mask, image_width, image_height, mask_value,
                format, normalized, fill_mode, border_width, min_bbox_size
            )
            
            # 创建可视化图像
            visualization = self.create_visualization(mask)
            
            # 创建信息字符串
            info = self.create_info_string(
                boxes, processed_boxes, image_width, image_height,
                format, normalized
            )
            
            # 创建像素坐标bbox列表
            pixel_bbox_list = self.create_pixel_bbox_list(processed_boxes)
            
            # 创建归一化坐标bbox列表
            normalized_bbox_list = self.create_normalized_bbox_list(processed_boxes, image_width, image_height)
            
            # 创建格式化的bbox字符串（XYXY格式，归一化）
            formatted_string = self.create_formatted_bbox_string(processed_boxes, image_width, image_height)
            
            return (mask, visualization, info, pixel_bbox_list, normalized_bbox_list, formatted_string)
            
        except Exception as e:
            print(f"Error converting bbox to mask: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回一个空的mask
            empty_mask = torch.zeros((1, image_height, image_width), dtype=torch.float32)
            empty_vis = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            info = f"Error: {str(e)}"
            formatted_string = '{"boxes": []}'
            return (empty_mask, empty_vis, info, [], [], formatted_string)
    
    def parse_bbox_string(self, bbox_string):
        """解析bbox字符串，提取boxes，忽略labels"""
        try:
            # 清理字符串
            bbox_string = bbox_string.strip()
            
            if debug_mode == "yes":
                print(f"Parsing bbox_string: {bbox_string[:100]}...")
            
            # 尝试JSON解析
            if bbox_string.startswith('{') or bbox_string.startswith('['):
                # 替换单引号为双引号以兼容JSON
                bbox_string = bbox_string.replace("'", '"')
                # 处理True/False
                bbox_string = bbox_string.replace('True', 'true').replace('False', 'false')
                
                # 尝试解析为JSON
                try:
                    data = json.loads(bbox_string)
                except json.JSONDecodeError:
                    # 尝试使用ast解析
                    data = ast.literal_eval(bbox_string)
                
                # 检查数据格式
                if isinstance(data, dict):
                    # 格式1: {'boxes': [[x1,y1,x2,y2],...], 'labels': [...]}
                    if 'boxes' in data:
                        return data['boxes']
                    # 格式2: {'bbox': [[x1,y1,x2,y2],...], 'label': [...]}
                    elif 'bbox' in data:
                        return data['bbox']
                    # 格式3: 字典中有坐标键
                    else:
                        # 查找可能的bbox键
                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) > 0:
                                if isinstance(data[key][0], list) and len(data[key][0]) >= 4:
                                    return data[key]
                elif isinstance(data, list):
                    # 格式4: 纯列表格式 [[x1,y1,x2,y3],...]
                    return data
            
            # 尝试解析为纯Python表达式
            try:
                data = ast.literal_eval(bbox_string)
                if isinstance(data, dict):
                    # 从字典中提取boxes
                    if 'boxes' in data:
                        return data['boxes']
                    elif 'bbox' in data:
                        return data['bbox']
                    else:
                        # 返回第一个列表类型的值
                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) > 0:
                                if isinstance(data[key][0], list) and len(data[key][0]) >= 4:
                                    return data[key]
                elif isinstance(data, list):
                    return data
            except:
                pass
            
            # 如果无法解析，尝试按行分割
            lines = bbox_string.strip().split('\n')
            boxes = []
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    try:
                        box = ast.literal_eval(line)
                        if isinstance(box, list) and len(box) >= 4:
                            boxes.append(box[:4])  # 只取前4个值
                    except:
                        pass
            
            if boxes:
                return boxes
            
            raise ValueError("无法解析bbox字符串格式")
            
        except Exception as e:
            raise ValueError(f"Failed to parse bbox string: {e}")
    
    def detect_format_and_normalization(self, boxes, width, height):
        """自动检测格式和归一化状态"""
        if not boxes:
            return "xyxy", "yes"
        
        # 采样前几个box进行检测
        sample_boxes = boxes[:min(5, len(boxes))]
        
        # 检测格式
        format_scores = {"xyxy": 0, "xywh": 0, "cxcywh": 0}
        
        for box in sample_boxes:
            if len(box) >= 4:
                val1, val2, val3, val4 = box[:4]
                
                # 如果是xyxy格式，通常x2 > x1, y2 > y1
                if val3 > val1 and val4 > val2:
                    format_scores["xyxy"] += 1
                
                # 如果是xywh格式，通常w和h是正数，且x+w和y+h可能在0-1或0-图像尺寸范围内
                # 检查是否可能是宽度和高度（正数）
                if val3 > 0 and val4 > 0:
                    # 对于归一化情况
                    if 0 <= val1 <= 1 and 0 <= val2 <= 1:
                        # 检查x+w和y+h是否在合理范围
                        if val1 + val3 <= 1.1 and val2 + val4 <= 1.1:
                            format_scores["xywh"] += 1
                    # 对于非归一化情况
                    elif val1 + val3 <= width * 1.1 and val2 + val4 <= height * 1.1:
                        format_scores["xywh"] += 1
                
                # 如果是cxcywh格式，中心点通常在图像内，宽高为正数
                # 检查是否可能是中心点坐标
                if val3 > 0 and val4 > 0:
                    # 检查中心点是否在合理范围内
                    # 对于归一化情况
                    if 0 <= val1 <= 1 and 0 <= val2 <= 1:
                        # 检查左上角坐标是否在合理范围
                        x1 = val1 - val3/2
                        y1 = val2 - val4/2
                        if -0.1 <= x1 <= 1.1 and -0.1 <= y1 <= 1.1:
                            format_scores["cxcywh"] += 1
                    # 对于非归一化情况
                    elif 0 <= val1 <= width and 0 <= val2 <= height:
                        x1 = val1 - val3/2
                        y1 = val2 - val4/2
                        if -width*0.1 <= x1 <= width*1.1 and -height*0.1 <= y1 <= height*1.1:
                            format_scores["cxcywh"] += 1
        
        # 选择分数高的格式
        detected_format = max(format_scores, key=format_scores.get)
        
        # 如果所有格式得分都为0，默认使用xyxy
        if format_scores[detected_format] == 0:
            detected_format = "xyxy"
        
        # 检测归一化状态
        normalized_scores = {"yes": 0, "no": 0}
        
        for box in sample_boxes:
            if len(box) >= 4:
                # 检查前两个值（通常是位置坐标）
                for val in box[:2]:
                    if 0 <= val <= 1:
                        normalized_scores["yes"] += 1
                    elif 0 <= val <= max(width, height) * 1.5:
                        normalized_scores["no"] += 1
        
        # 选择分数高的归一化状态
        if normalized_scores["yes"] >= normalized_scores["no"]:
            detected_normalized = "yes"
        else:
            detected_normalized = "no"
        
        return detected_format, detected_normalized
    
    def process_boxes(self, boxes, mask, width, height, mask_value, format, normalized, 
                     fill_mode, border_width, min_bbox_size):
        """处理所有bbox并更新mask"""
        processed_boxes = []
        
        for i, box in enumerate(boxes):
            # 检查box格式
            if len(box) < 4:
                print(f"Warning: BBox {i} has invalid format: {box}")
                continue
            
            try:
                # 根据格式处理坐标
                if format == "xyxy":
                    if normalized == "yes":
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]
                        
                        # 转换为像素坐标
                        x1_px = int(x1 * width)
                        y1_px = int(y1 * height)
                        x2_px = int(x2 * width)
                        y2_px = int(y2 * height)
                        
                        # 计算归一化宽度和高度（用于输出）
                        w_norm = x2 - x1
                        h_norm = y2 - y1
                    else:
                        x1_px = int(box[0])
                        y1_px = int(box[1])
                        x2_px = int(box[2])
                        y2_px = int(box[3])
                        
                        # 转换为归一化坐标
                        x1 = x1_px / width
                        y1 = y1_px / height
                        x2 = x2_px / width
                        y2 = y2_px / height
                        w_norm = x2 - x1
                        h_norm = y2 - y1
                    
                    # 确保x2 > x1, y2 > y1
                    x1_px, x2_px = min(x1_px, x2_px), max(x1_px, x2_px)
                    y1_px, y2_px = min(y1_px, y2_px), max(y1_px, y2_px)
                
                elif format == "xywh":
                    if normalized == "yes":
                        x = box[0]
                        y = box[1]
                        w_norm = box[2]
                        h_norm = box[3]
                        
                        # 转换为像素坐标
                        x1_px = int(x * width)
                        y1_px = int(y * height)
                        w_px = int(w_norm * width)
                        h_px = int(h_norm * height)
                        
                        x1 = x
                        y1 = y
                        x2 = x + w_norm
                        y2 = y + h_norm
                    else:
                        x1_px = int(box[0])
                        y1_px = int(box[1])
                        w_px = int(box[2])
                        h_px = int(box[3])
                        
                        # 转换为归一化坐标
                        x1 = x1_px / width
                        y1 = y1_px / height
                        w_norm = w_px / width
                        h_norm = h_px / height
                        x2 = x1 + w_norm
                        y2 = y1 + h_norm
                    
                    x2_px = x1_px + w_px
                    y2_px = y1_px + h_px
                
                elif format == "cxcywh":
                    if normalized == "yes":
                        cx = box[0]  # 中心点x
                        cy = box[1]  # 中心点y
                        w_norm = box[2]
                        h_norm = box[3]
                        
                        # 计算左上角坐标
                        x1 = cx - w_norm / 2
                        y1 = cy - h_norm / 2
                        x2 = cx + w_norm / 2
                        y2 = cy + h_norm / 2
                        
                        # 转换为像素坐标
                        x1_px = int(x1 * width)
                        y1_px = int(y1 * height)
                        x2_px = int(x2 * width)
                        y2_px = int(y2 * height)
                        w_px = int(w_norm * width)
                        h_px = int(h_norm * height)
                    else:
                        cx_px = int(box[0])
                        cy_px = int(box[1])
                        w_px = int(box[2])
                        h_px = int(box[3])
                        
                        # 计算像素坐标的左上角
                        x1_px = cx_px - w_px // 2
                        y1_px = cy_px - h_px // 2
                        x2_px = x1_px + w_px
                        y2_px = y1_px + h_px
                        
                        # 转换为归一化坐标
                        x1 = x1_px / width
                        y1 = y1_px / height
                        x2 = x2_px / width
                        y2 = y2_px / height
                        w_norm = w_px / width
                        h_norm = h_px / height
                
                # 确保坐标在图像范围内
                x1_px = max(0, min(x1_px, width - 1))
                x2_px = max(0, min(x2_px, width - 1))
                y1_px = max(0, min(y1_px, height - 1))
                y2_px = max(0, min(y2_px, height - 1))
                
                # 计算bbox尺寸
                bbox_width = x2_px - x1_px
                bbox_height = y2_px - y1_px
                
                # 检查最小尺寸
                if bbox_width < min_bbox_size or bbox_height < min_bbox_size:
                    print(f"Warning: BBox {i} is too small ({bbox_width}x{bbox_height}), skipping")
                    continue
                
                # 确保宽度和高度为正
                if x2_px <= x1_px or y2_px <= y1_px:
                    print(f"Warning: BBox {i} has invalid dimensions after clipping: {box}")
                    continue
                
                # 根据填充模式更新mask
                if border_width > 0:
                    # 绘制边框
                    self.draw_border(mask, x1_px, y1_px, x2_px, y2_px, mask_value, border_width)
                else:
                    # 填充整个区域
                    if fill_mode == "rectangle":
                        self.draw_rectangle(mask, x1_px, y1_px, x2_px, y2_px, mask_value)
                    elif fill_mode == "ellipse":
                        self.draw_ellipse(mask, x1_px, y1_px, x2_px, y2_px, mask_value)
                    elif fill_mode == "gaussian":
                        self.draw_gaussian(mask, x1_px, y1_px, x2_px, y2_px, mask_value)
                    elif fill_mode == "soft_rectangle":
                        self.draw_soft_rectangle(mask, x1_px, y1_px, x2_px, y2_px, mask_value)
                
                # 保存处理后的bbox
                processed_box = {
                    'index': i,
                    'format': format,
                    'x1_px': x1_px,
                    'y1_px': y1_px,
                    'x2_px': x2_px,
                    'y2_px': y2_px,
                    'width_px': bbox_width,
                    'height_px': bbox_height,
                    'x1_norm': x1,
                    'y1_norm': y1,
                    'x2_norm': x2,
                    'y2_norm': y2,
                    'width_norm': w_norm,
                    'height_norm': h_norm,
                }
                
                # 如果是cxcywh格式，保存中心点信息
                if format == "cxcywh":
                    if normalized == "yes":
                        processed_box['cx_norm'] = box[0]
                        processed_box['cy_norm'] = box[1]
                    else:
                        processed_box['cx_px'] = int(box[0])
                        processed_box['cy_px'] = int(box[1])
                        processed_box['cx_norm'] = processed_box['cx_px'] / width
                        processed_box['cy_norm'] = processed_box['cy_px'] / height
                
                processed_boxes.append(processed_box)
                
            except Exception as e:
                print(f"Error processing box {i} ({box}): {e}")
                continue
        
        return processed_boxes
    
    def draw_rectangle(self, mask, x1, y1, x2, y2, value):
        """绘制矩形"""
        mask[0, y1:y2, x1:x2] = value
    
    def draw_border(self, mask, x1, y1, x2, y2, value, border_width):
        """绘制边框"""
        # 确保border_width不超过bbox尺寸
        border_width = min(border_width, (x2 - x1) // 2, (y2 - y1) // 2)
        if border_width <= 0:
            return
        
        # 绘制上边
        mask[0, y1:min(y1+border_width, y2), x1:x2] = value
        # 绘制下边
        mask[0, max(y1, y2-border_width):y2, x1:x2] = value
        # 绘制左边
        mask[0, y1:y2, x1:min(x1+border_width, x2)] = value
        # 绘制右边
        mask[0, y1:y2, max(x1, x2-border_width):x2] = value
    
    def draw_ellipse(self, mask, x1, y1, x2, y2, value):
        """绘制椭圆"""
        h, w = mask.shape[1], mask.shape[2]
        
        # 创建网格
        y_indices, x_indices = torch.meshgrid(
            torch.arange(h, device=mask.device),
            torch.arange(w, device=mask.device),
            indexing='ij'
        )
        
        # 计算中心点和半径
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius_x = (x2 - x1) / 2
        radius_y = (y2 - y1) / 2
        
        # 避免除零错误
        if radius_x <= 0 or radius_y <= 0:
            return
        
        # 椭圆方程
        ellipse_mask = ((x_indices - center_x) / radius_x) ** 2 + \
                      ((y_indices - center_y) / radius_y) ** 2 <= 1
        
        # 应用椭圆mask
        mask[0][ellipse_mask] = value
    
    def draw_gaussian(self, mask, x1, y1, x2, y2, value):
        """绘制高斯分布"""
        h, w = mask.shape[1], mask.shape[2]
        
        # 创建网格
        y_indices, x_indices = torch.meshgrid(
            torch.arange(h, device=mask.device),
            torch.arange(w, device=mask.device),
            indexing='ij'
        )
        
        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 计算sigma为bbox尺寸的1/4
        sigma_x = max((x2 - x1) / 4, 1)
        sigma_y = max((y2 - y1) / 4, 1)
        
        # 计算高斯分布
        gaussian = torch.exp(
            -(((x_indices - center_x) / sigma_x) ** 2 + ((y_indices - center_y) / sigma_y) ** 2) / 2
        )
        
        # 限制在bbox范围内
        bbox_mask = (x_indices >= x1) & (x_indices < x2) & \
                   (y_indices >= y1) & (y_indices < y2)
        
        # 应用高斯mask
        gaussian_value = gaussian * value
        current_mask = mask[0].clone()
        current_mask[bbox_mask] = torch.maximum(current_mask[bbox_mask], gaussian_value[bbox_mask])
        mask[0] = current_mask
    
    def draw_soft_rectangle(self, mask, x1, y1, x2, y2, value):
        """绘制软矩形（带羽化边缘）"""
        h, w = mask.shape[1], mask.shape[2]
        
        # 创建网格
        y_indices, x_indices = torch.meshgrid(
            torch.arange(h, device=mask.device),
            torch.arange(w, device=mask.device),
            indexing='ij'
        )
        
        # 计算到边界的距离
        dist_left = (x_indices - x1).float()
        dist_right = (x2 - x_indices).float()
        dist_top = (y_indices - y1).float()
        dist_bottom = (y2 - y_indices).float()
        
        # 计算到最近边界的距离
        dist_x = torch.minimum(dist_left, dist_right)
        dist_y = torch.minimum(dist_top, dist_bottom)
        dist = torch.minimum(dist_x, dist_y)
        
        # 计算羽化宽度（bbox最小尺寸的20%）
        feather_width = max(min(x2 - x1, y2 - y1) * 0.2, 1)
        
        # 计算软mask
        soft_mask = torch.clamp(dist / feather_width, 0, 1)
        soft_mask = 1 - soft_mask  # 边缘为0，中心为1
        
        # 只在bbox范围内应用
        bbox_mask = (x_indices >= x1) & (x_indices < x2) & \
                   (y_indices >= y1) & (y_indices < y2)
        
        # 应用软矩形mask
        current_mask = mask[0].clone()
        new_mask = current_mask.clone()
        new_mask[bbox_mask] = torch.maximum(current_mask[bbox_mask], soft_mask[bbox_mask] * value)
        mask[0] = new_mask
    
    def create_visualization(self, mask):
        """创建可视化图像"""
        # mask形状: (1, H, W)
        batch_size, height, width = mask.shape
        
        # 转换为3通道图像用于可视化
        mask_3d = mask.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        mask_3d = mask_3d.permute(0, 2, 3, 1)  # (1, H, W, 3)
        
        return mask_3d
    
    def create_info_string(self, boxes, processed_boxes, width, height, format, normalized):
        """创建信息字符串"""
        total_boxes = len(boxes)
        processed_count = len(processed_boxes)
        
        info = f"Universal BBOX to Mask\n"
        info += "=" * 50 + "\n"
        info += f"Total boxes in input: {total_boxes}\n"
        info += f"Successfully processed: {processed_count}\n"
        info += f"Format detected/specified: {format}\n"
        info += f"Normalized detected/specified: {normalized}\n"
        info += f"Image size: {width}x{height}\n"
        
        if processed_boxes:
            info += "\nProcessed boxes (first 3):\n"
            info += "-" * 50 + "\n"
            for i, box in enumerate(processed_boxes[:3]):
                info += f"Box {box['index']}:\n"
                info += f"  Pixel: ({box['x1_px']}, {box['y1_px']}) to "
                info += f"({box['x2_px']}, {box['y2_px']})\n"
                info += f"  Size: {box['width_px']}x{box['height_px']}\n"
                info += f"  Normalized: ({box['x1_norm']:.4f}, {box['y1_norm']:.4f}) to "
                info += f"({box['x2_norm']:.4f}, {box['y2_norm']:.4f})\n"
                if 'cx_norm' in box:
                    info += f"  Center: ({box['cx_norm']:.4f}, {box['cy_norm']:.4f})\n"
                info += "\n"
            
            if len(processed_boxes) > 3:
                info += f"... and {len(processed_boxes) - 3} more boxes\n"
        
        info += "=" * 50 + "\n"
        return info
    
    def create_pixel_bbox_list(self, processed_boxes):
        """创建像素坐标bbox列表（XYXY格式）"""
        bbox_list = []
        for box in processed_boxes:
            bbox_list.append([
                box['x1_px'],
                box['y1_px'],
                box['x2_px'],
                box['y2_px']
            ])
        return bbox_list
    
    def create_normalized_bbox_list(self, processed_boxes, width, height):
        """创建归一化坐标bbox列表（XYXY格式）"""
        bbox_list = []
        for box in processed_boxes:
            bbox_list.append([
                box['x1_norm'],
                box['y1_norm'],
                box['x2_norm'],
                box['y2_norm']
            ])
        return bbox_list
    
    def create_formatted_bbox_string(self, processed_boxes, width, height):
        """创建格式化的bbox字符串（XYXY格式，归一化）"""
        if not processed_boxes:
            return '{"boxes": []}'
        
        boxes_list = []
        for box in processed_boxes:
            boxes_list.append([
                float(box['x1_norm']),
                float(box['y1_norm']),
                float(box['x2_norm']),
                float(box['y2_norm'])
            ])
        
        bbox_dict = {
            'boxes': boxes_list,
            'format': 'xyxy_normalized',
            'image_width': width,
            'image_height': height,
            'num_boxes': len(boxes_list)
        }
        
        return json.dumps(bbox_dict, indent=2)




class BBOXFormatConverterZV:
    """
    BBOX格式转换节点
    在不同格式之间转换
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_bbox": ("STRING", {
                    "multiline": True,
                    "default": "[[0.5, 0.5, 0.2, 0.3]]"
                }),
                "input_format": (["auto", "xyxy", "xywh", "cxcywh"], {
                    "default": "auto"
                }),
                "output_format": (["xyxy", "xywh", "cxcywh"], {
                    "default": "xyxy"
                }),
                "input_normalized": (["auto", "yes", "no"], {
                    "default": "auto"
                }),
                "output_normalized": (["yes", "no"], {
                    "default": "yes"
                }),
                "image_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "image_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "LIST", "STRING")
    RETURN_NAMES = ("output_bbox", "bbox_list", "info")
    FUNCTION = "convert_format"
    CATEGORY = "mask/bbox"
    
    def convert_format(self, input_bbox, input_format, output_format, 
                      input_normalized, output_normalized, image_width, image_height):
        info = ""
        try:
            # 解析输入bbox
            universal_node = UniversalBBOXToMask()
            boxes = universal_node.parse_bbox_string(input_bbox)
            
            if not boxes:
                return ('{"boxes": []}', [], "No boxes found")
            
            # 如果输入格式是auto，检测格式
            if input_format == "auto" or input_normalized == "auto":
                detected_format, detected_normalized = universal_node.detect_format_and_normalization(
                    boxes, image_width, image_height
                )
                if input_format == "auto":
                    input_format = detected_format
                if input_normalized == "auto":
                    input_normalized = detected_normalized
            
            info += f"Input format: {input_format}, normalized: {input_normalized}\n"
            info += f"Output format: {output_format}, normalized: {output_normalized}\n"
            
            # 转换每个bbox
            converted_boxes = []
            bbox_list = []
            
            for i, box in enumerate(boxes):
                if len(box) < 4:
                    continue
                
                # 转换为中间格式（像素坐标）
                if input_normalized == "yes":
                    if input_format == "xyxy":
                        x1_norm, y1_norm, x2_norm, y2_norm = box[:4]
                        x1 = x1_norm * image_width
                        y1 = y1_norm * image_height
                        x2 = x2_norm * image_width
                        y2 = y2_norm * image_height
                        w = x2 - x1
                        h = y2 - y1
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                    elif input_format == "xywh":
                        x_norm, y_norm, w_norm, h_norm = box[:4]
                        x1 = x_norm * image_width
                        y1 = y_norm * image_height
                        w = w_norm * image_width
                        h = h_norm * image_height
                        x2 = x1 + w
                        y2 = y1 + h
                        cx = x1 + w/2
                        cy = y1 + h/2
                    elif input_format == "cxcywh":
                        cx_norm, cy_norm, w_norm, h_norm = box[:4]
                        cx = cx_norm * image_width
                        cy = cy_norm * image_height
                        w = w_norm * image_width
                        h = h_norm * image_height
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = x1 + w
                        y2 = y1 + h
                else:
                    if input_format == "xyxy":
                        x1, y1, x2, y2 = map(float, box[:4])
                        w = x2 - x1
                        h = y2 - y1
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                    elif input_format == "xywh":
                        x1, y1, w, h = map(float, box[:4])
                        x2 = x1 + w
                        y2 = y1 + h
                        cx = x1 + w/2
                        cy = y1 + h/2
                    elif input_format == "cxcywh":
                        cx, cy, w, h = map(float, box[:4])
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = x1 + w
                        y2 = y1 + h
                
                # 转换为输出格式
                if output_normalized == "yes":
                    if output_format == "xyxy":
                        out_box = [
                            x1 / image_width,
                            y1 / image_height,
                            x2 / image_width,
                            y2 / image_height
                        ]
                    elif output_format == "xywh":
                        out_box = [
                            x1 / image_width,
                            y1 / image_height,
                            w / image_width,
                            h / image_height
                        ]
                    elif output_format == "cxcywh":
                        out_box = [
                            cx / image_width,
                            cy / image_height,
                            w / image_width,
                            h / image_height
                        ]
                else:
                    if output_format == "xyxy":
                        out_box = [x1, y1, x2, y2]
                    elif output_format == "xywh":
                        out_box = [x1, y1, w, h]
                    elif output_format == "cxcywh":
                        out_box = [cx, cy, w, h]
                
                # 添加到结果列表
                converted_boxes.append(out_box)
                bbox_list.append(out_box)
            
            # 创建输出字符串
            output_dict = {
                'boxes': converted_boxes,
                'input_format': input_format,
                'output_format': output_format,
                'input_normalized': input_normalized,
                'output_normalized': output_normalized,
                'image_size': f"{image_width}x{image_height}",
                'num_boxes': len(converted_boxes)
            }
            
            output_str = json.dumps(output_dict, indent=2)
            info += f"Successfully converted {len(converted_boxes)} boxes\n"
            
            return (output_str, bbox_list, info)
            
        except Exception as e:
            error_msg = f"Error in format conversion: {str(e)}"
            print(error_msg)
            return ('{"boxes": []}', [], error_msg)


def save_image(img: torch.Tensor, path, quality, prompt=None, extra_pnginfo: dict = None):
    path = str(path)

    if len(img.shape) != 3:
        raise ValueError(f"can't take image batch as input, got {img.shape[0]} images")

    img = img.permute(2, 0, 1)
    if img.shape[0] not in (3, 4):
        raise ValueError(
            f"image must have 3 or 4 channels, but got {img.shape[0]} channels"
        )

    img = img.clamp(0, 1)
    img = F.to_pil_image(img)

    metadata = PngInfo()

    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata.add_text(k, json.dumps(v))

    

    subfolder, filename = os.path.split(path)
    ext = os.path.splitext(filename)[-1].lower()
    # 根据格式保存图像
    save_params = {}
    if ext == ".png":
        save_params["pnginfo"]=metadata
        save_params["compress_level"] = 9 - min(9, max(0, quality // 10))
    elif ext == ".jpg":
        img = img.convert("RGB")
        save_params["quality"] = quality
        save_params["subsampling"] = 0
    elif ext == ".webp":
        save_params["quality"] = quality
    elif ext == ".tiff":
        save_params["compression"] = "tiff_deflate"
    elif ext == ".heic":
        save_params["quality"] = quality
    
    img.save(path, **save_params)

    return {"filename": filename, "subfolder": subfolder, "type": "output"}


class ImageCounterNodeZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "folder_picker": True}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("image_count",)
    FUNCTION = "count_images"
    CATEGORY = "ZVNodes/image"
    DESCRIPTION = "Count images in a directory"

    def count_images(self, directory, include_subfolders):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # 支持的图片扩展名
        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".pjp", ".pjpeg", ".jfif"]
        
        count = 0
        # 更高效的文件遍历方式
        if include_subfolders:
            # 递归遍历所有文件和子目录
            for _, _, files in os.walk(directory):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        else:
            # 仅检查顶层目录
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)):
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        
        return (count,)


class LoadImageFromDirZV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""})
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "directory", "name", "prefix")
    FUNCTION = "load_images"
    CATEGORY = "ZVNodes/image"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

    def load_images(self, folder, start_index, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".pjp", ".pjpeg", ".jfif"]
        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions) and os.path.isfile(os.path.join(folder, file)):
                    image_paths.append(os.path.join(folder, file))

        dir_files = sorted(image_paths)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        if len(dir_files) > start_index:
            image_path = dir_files[start_index]
        else:
            raise FileNotFoundError(f"No Enough files in directory '{folder}'.")

        i = Image.open(image_path)
        width, height = i.size
        i = ImageOps.exif_transpose(i)
        
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            if mask.shape != (height, width):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                        size=(height, width), 
                                                        mode='bilinear', 
                                                        align_corners=False).squeeze()
        else:
            mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        
        image_dir, imagename = os.path.split(image_path)
        image_prefix= os.path.splitext(imagename)[0].lower()
        image_dir  = os.path.relpath(image_dir, folder)

        return (image, mask, image_dir, imagename, image_prefix)

class SaveImageToPathZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "folder": ("STRING", {"default": "."}),
                        "subfolder": ("STRING", {"default": "."}),
                        "prefix": ("STRING", {"default": "image"}),
                        "file_extension": ((".png", ".jpg", ".webp", ".tiff", ".bmp", ".heic"), {"default": ".png"}),
                        "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                        "image": ("IMAGE",),
                        "metadata": ("BOOLEAN", {"default": False}),
                        "storage_method": (["folder_based", "suffix_based"], {"default": "folder_based"}),
                        "num_padding": ("INT", {"default": 4, "min": 0, "step": 1}),
                        "overwrite": ("BOOLEAN", {"default": True}),
                    },
                    "optional": {
                        "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                        "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}), 
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    CATEGORY = "ZVNodes/image"
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(
        self,
        folder: str,
        subfolder: str,
        prefix: str,
        file_extension: str,
        quality: int,
        image: torch.Tensor,
        metadata: bool,
        storage_method: str,
        num_padding: int,
        overwrite: bool,
        caption=None, 
        caption_file_extension=".txt",
        prompt=None,
        extra_pnginfo=None,
    ):
        assert isinstance(folder, str)
        assert isinstance(subfolder, str)
        assert isinstance(prefix, str)
        assert isinstance(file_extension, str)
        assert isinstance(quality, int)
        assert isinstance(image, torch.Tensor)
        assert isinstance(metadata, bool)
        assert isinstance(storage_method, str)
        assert isinstance(num_padding, int)
        assert isinstance(overwrite, bool)

        image_path = os.path.join(folder,subfolder,f"{prefix}{file_extension}")
        path: Path = Path(image_path)
        image_path_list = []
        results = []
        if not overwrite and path.exists():
            return (image_path_list,)

        path.parent.mkdir(exist_ok=True, parents=True)

        if metadata:
            _prompt=prompt
            _extra_pnginfo=extra_pnginfo
        else:
            _prompt=None
            _extra_pnginfo=None
        
        

        if image.shape[0] == 1:
            # batch has 1 image only
            results.append(
                save_image(
                    image[0],
                    path,
                    quality,
                    prompt=_prompt,
                    extra_pnginfo=_extra_pnginfo,
                )
            )
            if caption is not None:
                txt_path = path.parent / (path.stem+caption_file_extension)
                with txt_path.open('w', encoding="UTF-8") as f:
                    f.write(caption)
            image_path_list.append(str(path))
        else:
            # batch has multiple images
            for i, img in enumerate(image):
                batch_name = str(i).zfill(num_padding)
                if storage_method == "suffix_based":
                    subpath = path.with_stem(f"{path.stem}_{batch_name}")
                else:
                    subpath = path.parent / batch_name / path.name
                    subpath.parent.mkdir(exist_ok=True, parents=True)
                    
                results.append(
                    save_image(
                        img,
                        subpath,
                        quality,
                        prompt=_prompt,
                        extra_pnginfo=_extra_pnginfo,
                    )
                )
                if caption is not None:
                    txt_path = subpath.parent / (subpath.stem+caption_file_extension)
                    with txt_path.open('w', encoding="UTF-8") as f:
                        f.write(caption)
                image_path_list.append(str(subpath))

        return (image_path_list,)

class LoadImageFromUrlZV:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seededit_i2i.jpeg"
                    },
                )
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING")
    RETURN_NAMES = ("image", "image_url", "filename", "image_prefix")
    FUNCTION = "load"
    CATEGORY = "ZVNodes/image"

    def load(self, url):
        # get the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        image_prefix = os.path.splitext(filename)[0]
        return (pil2tensor(image), url, filename, image_prefix)
    

NODE_CONFIG = {
    "LoadImageFromUrlZV":{"class": LoadImageFromUrlZV, "name": "Load Image (Url)"},
    "LoadImageFromDirZV": {"class": LoadImageFromDirZV, "name": "Load One Image (Directory)"},
    "SaveImageToPathZV": {"class": SaveImageToPathZV, "name": "Save Image (Directory)"},
    "ImageCounterNodeZV":{"class": ImageCounterNodeZV, "name": "Count Image (Directory)"},
    "UniversalBBOXToMaskZV": {"class": UniversalBBOXToMaskZV, "name": "Universal BBOX To Mask"},
    "BBOXFormatConverterZV":  {"class": BBOXFormatConverterZV, "name": "BBOX Format Converter"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)