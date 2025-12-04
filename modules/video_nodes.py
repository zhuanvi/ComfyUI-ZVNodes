import os
import torch
import numpy as np
from PIL import Image
import json
import torch
import cv2
from datetime import datetime
import folder_paths
import tempfile
import subprocess
from comfy.comfy_types import IO
from comfy_api.input import VideoInput
from comfy_api.util import VideoContainer, VideoCodec, VideoComponents
from scenedetect import SceneManager, VideoManager, ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode

from .utils import generate_node_mappings, calculate_file_hash

class LoadVideoFromDirZV:
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "video_name")
    FUNCTION = "get_video_info"
    CATEGORY = "ZVNodes/video"
    DESCRIPTION = """Get video file path and name from a folder by index."""

    def get_video_info(self, folder, start_index=0, include_subfolders=False):
        # 检查文件夹是否存在
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        # 支持的视频格式
        valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpg", ".mpeg", ".m4v"]
        video_paths = []
        
        # 收集视频文件
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        video_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions) and os.path.isfile(os.path.join(folder, file)):
                    video_paths.append(os.path.join(folder, file))

        # 排序
        sorted_videos = sorted(video_paths)

        # 检查是否有视频文件
        if len(sorted_videos) == 0:
            raise FileNotFoundError(f"No video files found in directory '{folder}'.")
        
        # 检查索引是否有效
        if start_index >= len(sorted_videos):
            raise ValueError(f"Video index {start_index} out of range. Only {len(sorted_videos)} videos found.")
        
        # 获取视频路径和名称
        video_path = sorted_videos[start_index]
        video_name = os.path.basename(video_path)
        
        return (video_path, video_name)

    @classmethod
    def IS_CHANGED(cls, folder, start_index=0, include_subfolders=False):
        # 当文件夹内容发生变化时，节点会重新执行
        return os.path.getmtime(folder) if os.path.exists(folder) else None

class VideoSpeedZV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "speed_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "interpolation": (["nearest", "linear", "cubic"], {"default": "linear"}),
                "frame_handling": (["drop", "duplicate", "blend"], {"default": "blend"}),
            },
            "optional": {
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "end_frame": ("INT", {"default": -1, "min": -1}),
                "target_frame_count": ("INT", {"default": -1, "min": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("output_images", "output_frame_count")
    FUNCTION = "process_image_batch"
    CATEGORY = "ZVNodes/video"
    
    def tensor_to_pil(self, tensor):
        """将PyTorch张量转换为PIL图像"""
        tensor = tensor.clone().detach().cpu()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        images = []
        for i in range(tensor.shape[0]):
            img_np = tensor[i].numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            if img_np.shape[2] == 4:  # RGBA
                image = Image.fromarray(img_np, 'RGBA')
            elif img_np.shape[2] == 3:  # RGB
                image = Image.fromarray(img_np, 'RGB')
            elif img_np.shape[2] == 1:  # Grayscale
                image = Image.fromarray(img_np[:, :, 0], 'L')
            else:
                raise ValueError(f"Unsupported number of channels: {img_np.shape[2]}")
                
            images.append(image)
        return images

    def pil_to_tensor(self, images):
        """将PIL图像列表转换为PyTorch张量"""
        tensors = []
        for img in images:
            if img.mode == 'RGBA':
                img_np = np.array(img).astype(np.float32) / 255.0
            else:
                img = img.convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(img_np))
        
        return torch.stack(tensors)

    def process_image_batch(self, images, speed_factor, interpolation, frame_handling, 
                          start_frame=0, end_frame=-1, target_frame_count=-1):
        # 验证输入
        if images.dim() != 4:
            raise ValueError("输入图像批次必须是4维张量 [batch, height, width, channels]")
            
        batch_size, height, width, channels = images.shape
        total_frames = batch_size
        
        # 处理帧范围
        if end_frame < 0 or end_frame >= total_frames:
            end_frame = total_frames - 1
        else:
            end_frame = min(end_frame, total_frames - 1)
            
        start_frame = max(0, min(start_frame, total_frames - 1))
        
        if start_frame > end_frame:
            raise ValueError("起始帧不能大于结束帧")
            
        # 计算实际处理的帧数
        frame_count = end_frame - start_frame + 1
        selected_images = images[start_frame:end_frame + 1]
        
        # 计算目标帧数
        if target_frame_count > 0:
            output_frame_count = target_frame_count
        else:
            output_frame_count = max(1, min(10000, int(round(frame_count / speed_factor))))
        
        print(f"输入帧数: {frame_count}, 速度因子: {speed_factor}, 输出帧数: {output_frame_count}")
        
        # 转换张量为PIL图像列表以便处理
        input_images = self.tensor_to_pil(selected_images)
        
        # 根据选择的处理方式创建输出图像序列
        output_images = []
        
        if frame_handling == "drop":
            # 简单丢弃帧
            step = frame_count / output_frame_count
            for i in range(output_frame_count):
                idx = min(frame_count - 1, int(i * step))
                output_images.append(input_images[idx])
        
        elif frame_handling == "duplicate":
            # 复制帧
            step = frame_count / output_frame_count
            for i in range(output_frame_count):
                idx = min(frame_count - 1, int(i * step))
                output_images.append(input_images[idx])
        
        elif frame_handling == "blend":
            # 帧混合（运动模糊效果）
            step = (frame_count - 1) / (output_frame_count - 1) if output_frame_count > 1 else 0
            
            for i in range(output_frame_count):
                pos = i * step
                idx1 = min(frame_count - 1, int(pos))
                idx2 = min(frame_count - 1, idx1 + 1)
                
                # 如果是整数位置，直接取帧
                if idx1 == idx2 or idx2 >= frame_count:
                    output_images.append(input_images[idx1])
                    continue
                
                # 计算混合权重
                weight = pos - idx1
                
                # 混合两帧
                img1 = np.array(input_images[idx1]).astype(np.float32)
                img2 = np.array(input_images[idx2]).astype(np.float32)
                
                # 确保图像尺寸相同
                if img1.shape != img2.shape:
                    img2 = np.array(input_images[idx2].resize(input_images[idx1].size, 
                                                             getattr(Image, interpolation.upper())))
                
                blended = (img1 * (1 - weight) + img2 * weight).astype(np.uint8)
                output_images.append(Image.fromarray(blended))
        
        # 转换为张量输出
        output_tensor = self.pil_to_tensor(output_images)
        
        return (output_tensor, output_tensor.shape[0])
    
class VideoSceneDetectorZV:
    """
    视频分镜切割节点 - 支持多种场景检测方法
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, ),
                "detector_type": (["content", "threshold", "adaptive"], {"default": "content"}),
            },
            "optional": {
                "content_threshold": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "min_scene_len": ("INT", {"default": 15, "min": 1, "max": 1000, "step": 1}),
                "threshold_value": ("INT", {"default": 12, "min": 0, "max": 255, "step": 1}),
                "adaptive_threshold": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "JSON")
    RETURN_NAMES = ("视频路径列表", "信息", "JSON数据")
    FUNCTION = "detect_scenes"
    CATEGORY = "ZVNodes/video"

    def detect_scenes(self, video: VideoInput, detector_type, **kwargs):
        
        output_prefix = "scene"
        # 处理临时视频文件
        video_path = video.get_stream_source()
        temp_flag = False
        if not isinstance(video_path, str):
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            video_path = temp_video.name
            video.save_to(
                temp_video,
                format=VideoContainer.MP4,
                codec=VideoCodec.H264
            )
            temp_flag = True

        # 验证视频路径
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 获取原视频文件名（不带扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 创建输出目录 - 使用原视频文件名作为前缀
        output_dir = folder_paths.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 修改文件夹名称为: [原视频名]_[时间戳]
        scene_dir = os.path.join(output_dir, f"{video_name}_scenes_{timestamp}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # 初始化视频和场景管理器
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        # 根据选择的检测器类型创建对应的检测器
        if detector_type == "content":
            detector = ContentDetector(
                threshold=kwargs.get("content_threshold", 30.0),
                min_scene_len=kwargs.get("min_scene_len", 15)
            )
        elif detector_type == "threshold":
            detector = ThresholdDetector(
                threshold=kwargs.get("threshold_value", 12),
                min_scene_len=kwargs.get("min_scene_len", 15)
            )
        elif detector_type == "adaptive":
            detector = AdaptiveDetector(
                adaptive_threshold=kwargs.get("adaptive_threshold", 3.0),
                min_scene_len=kwargs.get("min_scene_len", 15)
            )
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")
        
        scene_manager.add_detector(detector)
        
        # 设置视频降采样
        video_manager.set_downscale_factor()
        video_manager.start()
        
        # 执行场景检测
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        # 处理未检测到场景的情况
        if not scene_list:
            fps = video_manager.get_framerate()
            scene_list = [
                (FrameTimecode(timecode=0, fps=fps), 
                FrameTimecode(timecode=video_manager.frame_number, fps=fps))
            ]
        
        # 生成输出文件名模板 - 保持原视频名作为前缀
        output_template = os.path.join(scene_dir, f"{video_name}_{output_prefix}_$SCENE_NUMBER.mp4")
        
        # 分割视频
        split_video_ffmpeg(
            [video_path],
            scene_list,
            output_file_template=output_template,
            video_name=video_name,
            arg_override="-c copy"
        )

        # 清理临时文件
        if temp_flag:
            temp_video.close()
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
        
        # 收集场景信息
        scene_paths = []
        scene_info = []
        
        for i, scene in enumerate(scene_list):
            scene_number = i + 1
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time

            scene_path = output_template.replace("$SCENE_NUMBER", f"{scene_number:03d}")
            scene_paths.append(scene_path)
            
            scene_info.append({
                "scene_number": scene_number,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "path": scene_path
            })
        
        # 生成JSON输出
        json_data = json.dumps({
            "original_video": video_path,
            "detector_type": detector_type,
            "detector_params": kwargs,
            "total_scenes": len(scene_list),
            "scenes": scene_info,
            "output_directory": scene_dir  # 添加输出目录信息
        }, indent=2)
        
        # 生成信息字符串 - 包含新的目录结构信息
        info_str = (f"检测器类型: {detector_type}\n"
                   f"检测到 {len(scene_list)} 个场景\n"
                   f"输出目录: {scene_dir}\n"
                   f"文件名模式: {video_name}_{output_prefix}_###.mp4\n"
                   f"参数: {', '.join([f'{k}={v}' for k, v in kwargs.items()])}")
        
        return (scene_paths, info_str, [json_data])


    
NODE_CONFIG = {
    "VideoSpeedZV": {"class": VideoSpeedZV, "name": "Video Speed"},
    "VideoSceneDetectorZV": {"class": VideoSceneDetectorZV, "name": "Video Scene Detector"},
    "LoadVideoFromDirZV":{"class": LoadVideoFromDirZV, "name": "Load One Video (Directory)"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)