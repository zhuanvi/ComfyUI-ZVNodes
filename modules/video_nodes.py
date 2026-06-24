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
from comfy_api.latest import io, Types, Input, ui
from comfy_api.input import VideoInput
from comfy_api.util import VideoContainer, VideoCodec, VideoComponents
from comfy.cli_args import args
from scenedetect import SceneManager, VideoManager, ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode
import random
import math
import re
import glob
import time

from .utils import generate_node_mappings, calculate_file_hash, get_temp_video_path, run_ffmpeg, save_tensor_images

class VideoCounterNodeZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "folder_picker": True}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                # 添加随机种子，用于绕过懒加载缓存
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("video_count",)
    FUNCTION = "count_videos"
    CATEGORY = "ZVNodes/video"
    DESCRIPTION = "Count videos in a directory (seed input forces refresh)"

    def count_videos(self, directory, include_subfolders, seed):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # 支持的视频扩展名
        extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv"]
        
        count = 0
        if include_subfolders:
            for _, _, files in os.walk(directory):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        else:
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)):
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        
        return (count,)

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
        valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webp",".webm", ".flv", ".wmv", ".mpg", ".mpeg", ".m4v"]
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


class SaveVideoToPathZV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoToPathZV",
            search_aliases=["export video"],
            display_name="Save Video (Directory)",
            category="video",
            essentials_category="Basics",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("folder", default=folder_paths.get_output_directory(), tooltip="The folder for the file to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, folder, filename_prefix, format: str, codec):
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder,
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )

        return

ALL_XFADE_TRANSITIONS = [
    "fade", "fadefast", "fadeslow", "fadeblack", "fadewhite", "fadegrays",
    "dissolve", "pixelize", "hblur", "distance",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "wipetl", "wipetr", "wipebl", "wipebr",
    "slideleft", "slideright", "slideup", "slidedown",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circlecrop", "rectcrop", "circleopen", "circleclose",
    "vertopen", "vertclose", "horzopen", "horzclose",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hlwind", "hrwind", "vuwind", "vdwind",
    "radial", "zoomin",
    "squeezeh", "squeezev",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
]

class FFmpegImageSlideShowZV:
    @classmethod
    def INPUT_TYPES(cls):
        transitions = ["random", "custom_random"] + ALL_XFADE_TRANSITIONS
        return {
            "required": {
                "image_folder": ("STRING", {"default": "", "multiline": False}),
                "output_folder": ("STRING", {"default": "", "multiline": False}),
                "frame_duration": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "transition_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "transition_type": (transitions,),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192}),
                "ken_burns_mode": (["disabled", "all", "custom"], {"default": "disabled"}),
                "dynamic_indices": ("STRING", {"default": "0,2", "multiline": False}),
                "zoom_range_min": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "zoom_range_max": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 4.0, "step": 0.01}),
                "pan_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "custom_transitions": ("STRING", {
                    "default": "fade,wipeleft,wiperight,circlecrop,rectcrop,dissolve",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "create_slideshow"
    OUTPUT_NODE = True
    CATEGORY = "FFmpeg/Animation"

    def create_slideshow(self, image_folder, output_folder, frame_duration, transition_duration,
                         transition_type, fps, width, height, ken_burns_mode, dynamic_indices,
                         zoom_range_min, zoom_range_max, pan_strength, custom_transitions=""):
        # ==================== 0. 扫描文件夹 ====================
        if not os.path.isdir(image_folder):
            raise ValueError(f"无效的输入文件夹路径: {image_folder}")

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'([0-9]+)', os.path.basename(s))]

        valid_img_ext = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif')
        img_files = sorted(
            [os.path.join(image_folder, f) for f in os.listdir(image_folder)
             if f.lower().endswith(valid_img_ext)],
            key=natural_sort_key
        )

        if len(img_files) < 2:
            raise ValueError(f"文件夹中至少需要 2 张图片，当前找到 {len(img_files)} 张。")

        batch_size = len(img_files)
        print(f"[FFmpegImageSlideShow] 扫描到图片数量：{batch_size}")

        # 扫描音频文件（只取第一个）
        valid_audio_ext = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus')
        audio_files = []
        for f in os.listdir(image_folder):
            if f.lower().endswith(valid_audio_ext):
                audio_files.append(os.path.join(image_folder, f))
                break

        # ==================== 0.5 图片预处理：统一转换为标准 RGB PNG ====================
        temp_img_dir = os.path.join(folder_paths.get_temp_directory(), "slideshow_processed")
        os.makedirs(temp_img_dir, exist_ok=True)
        processed_paths = []

        for idx, src_path in enumerate(img_files):
            try:
                dst_path = os.path.join(temp_img_dir, f"processed_{idx:04d}.png")
                with Image.open(src_path) as img:
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dst_path, 'PNG', compress_level=3)
                processed_paths.append(os.path.abspath(dst_path))
                print(f"[FFmpegImageSlideShow] 预处理图片 {idx}: {os.path.basename(src_path)} -> {dst_path}")
            except Exception as e:
                print(f"[Warning] 图片预处理失败 {src_path}: {e}，尝试直接使用原图")
                processed_paths.append(os.path.abspath(src_path))

        # ==================== 1. Ken Burns 索引解析 ====================
        use_ken_burns = [False] * batch_size
        if ken_burns_mode == "all":
            use_ken_burns = [True] * batch_size
        elif ken_burns_mode == "custom":
            try:
                indices = [int(x.strip()) for x in dynamic_indices.split(",") if x.strip() != ""]
                for idx in indices:
                    if 0 <= idx < batch_size:
                        use_ken_burns[idx] = True
            except Exception:
                print("[Warning] dynamic_indices 格式错误，所有图片将使用静态效果。")
        print(f"[FFmpegImageSlideShow] 动态效果索引：{[i for i, v in enumerate(use_ken_burns) if v]}")

        # ==================== 2. 精确帧数与总时长计算 ====================
        frame_d = max(1, int(round(frame_duration * fps)))
        trans_d = max(0, int(round(transition_duration * fps)))
        transition_dur_sec = trans_d / fps if trans_d > 0 else 0.0

        frame_counts = []
        for idx in range(batch_size):
            if idx == 0:
                fc = frame_d + trans_d
            elif idx == batch_size - 1:
                fc = frame_d
            else:
                fc = frame_d + trans_d
            frame_counts.append(max(1, fc))

        total_video_frames = sum(frame_counts) - (batch_size - 1) * trans_d
        video_duration = total_video_frames / fps
        print(f"[FFmpegImageSlideShow] 视频总时长: {video_duration:.3f}s, 总帧数: {total_video_frames}")

        # ==================== 3. 生成每个图片的视频片段 ====================
        clip_paths = []
        for idx, img_path in enumerate(processed_paths):
            clip_path = get_temp_video_path(prefix=f"clip_{idx}")
            total_frames = frame_counts[idx]

            if use_ken_burns[idx]:
                if total_frames < 2:
                    total_frames = 2

                start_zoom = random.uniform(zoom_range_min, zoom_range_max)
                end_zoom   = random.uniform(zoom_range_min, zoom_range_max)
                start_pan_x = random.uniform(-pan_strength, pan_strength) * width
                start_pan_y = random.uniform(-pan_strength, pan_strength) * height
                end_pan_x   = random.uniform(-pan_strength, pan_strength) * width
                end_pan_y   = random.uniform(-pan_strength, pan_strength) * height

                denom = max(1, total_frames - 1)
                ease_expr = f"(0.5-0.5*cos(PI*on/{denom}))"
                zoom_expr = f"{start_zoom:.6f}+({end_zoom:.6f}-{start_zoom:.6f})*{ease_expr}"
                zoom_cur = f"{start_zoom:.6f}+({end_zoom:.6f}-{start_zoom:.6f})*(0.5-0.5*cos(PI*on/{denom}))"
                x_expr = f"{start_pan_x:.2f}+({end_pan_x:.2f}-{start_pan_x:.2f})*{ease_expr}+iw/2-iw/(2*{zoom_cur})"
                y_expr = f"{start_pan_y:.2f}+({end_pan_y:.2f}-{start_pan_y:.2f})*{ease_expr}+ih/2-ih/(2*{zoom_cur})"

                filter_str = (
                    f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
                    f":d={total_frames}:s={width}x{height}:fps={fps}"
                )

                cmd = [
                    "ffmpeg", "-y",
                    "-i", img_path,
                    "-filter_complex", filter_str,
                    "-frames:v", str(total_frames),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "medium",
                    "-an",
                    clip_path
                ]
            else:
                vf = (
                    f"loop=loop=-1:size=1:start=0,"
                    f"scale={width}:{height}:flags=lanczos,"
                    f"fps={fps},"
                    f"format=pix_fmts=yuv420p"
                )
                cmd = [
                    "ffmpeg", "-y",
                    "-i", img_path,
                    "-vf", vf,
                    "-frames:v", str(total_frames),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "medium",
                    "-an",
                    clip_path
                ]

            run_ffmpeg(cmd, f"Generate clip {idx}")
            clip_paths.append(clip_path)

        # ==================== 4. 音频处理 ====================
        real_transitions = ALL_XFADE_TRANSITIONS

        audio_filter_str = ""
        audio_input_count = 0
        use_audio = False

        if audio_files:
            audio_path = audio_files[0]
            audio_dur = None

            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                    capture_output=True, text=True, timeout=15
                )
                audio_dur = float(result.stdout.strip())
                print(f"[FFmpegImageSlideShow] 检测到音频: {os.path.basename(audio_path)}, 时长: {audio_dur:.3f}s")
            except Exception as e:
                print(f"[Warning] ffprobe 探测失败: {e}，将使用 aloop 简单循环方案")

            if audio_dur and audio_dur > 0:
                use_audio = True
                crossfade = min(0.5, audio_dur * 0.4)

                if audio_dur >= video_duration:
                    audio_input_count = 1
                    aidx = batch_size
                    fade_out_start = max(0, video_duration - 1.0)
                    audio_filter_str = (
                        f"[{aidx}:a]atrim=start=0:end={video_duration:.6f},"
                        f"afade=t=in:st=0:d=0.5,"
                        f"afade=t=out:st={fade_out_start:.6f}:d=1.0[aout];"
                    )
                else:
                    if audio_dur > crossfade:
                        n_loops = math.ceil((video_duration - crossfade) / (audio_dur - crossfade))
                    else:
                        n_loops = max(2, int(math.ceil(video_duration / max(audio_dur, 0.1))))
                    n_loops = min(n_loops, 100)

                    audio_input_count = n_loops
                    print(f"[FFmpegImageSlideShow] 音频将循环 {n_loops} 次 (crossfade={crossfade:.2f}s)")

                    audio_filters = []
                    base_idx = batch_size
                    for i in range(n_loops):
                        audio_filters.append(
                            f"[{base_idx + i}:a]atrim=start=0:end={audio_dur:.6f}[a{i}];"
                        )

                    cur_label = "a0"
                    for i in range(1, n_loops):
                        out_label = f"ac{i}"
                        audio_filters.append(
                            f"[{cur_label}][a{i}]acrossfade=d={crossfade:.3f}[{out_label}];"
                        )
                        cur_label = out_label

                    fade_out_start = max(0, video_duration - 1.0)
                    audio_filters.append(
                        f"[{cur_label}]atrim=start=0:end={video_duration:.6f},"
                        f"afade=t=in:st=0:d={crossfade:.3f},"
                        f"afade=t=out:st={fade_out_start:.6f}:d=1.0[aout];"
                    )
                    audio_filter_str = "".join(audio_filters)
            else:
                use_audio = True
                audio_input_count = 1
                aidx = batch_size
                fade_out_start = max(0, video_duration - 1.0)
                audio_filter_str = (
                    f"[{aidx}:a]aloop=loop=-1:size=0,"
                    f"atrim=start=0:end={video_duration:.6f},"
                    f"afade=t=in:st=0:d=0.5,"
                    f"afade=t=out:st={fade_out_start:.6f}:d=1.0[aout];"
                )
                print("[FFmpegImageSlideShow] 使用 aloop 简单循环音频")

        # ==================== 5. 拼接视频片段 ====================
        inputs = []
        for p in clip_paths:
            inputs += ["-i", p]

        if use_audio:
            for _ in range(audio_input_count):
                inputs += ["-i", audio_files[0]]

        # 解析自定义随机池
        custom_pool = []
        if transition_type == "custom_random" and custom_transitions.strip():
            custom_pool = [t.strip() for t in custom_transitions.split(",") if t.strip()]
            # 过滤掉无效的转场
            custom_pool = [t for t in custom_pool if t in ALL_XFADE_TRANSITIONS]
            if not custom_pool:
                print("[Warning] custom_transitions 无有效转场，回退到全部随机")
                custom_pool = ALL_XFADE_TRANSITIONS
        elif transition_type == "custom_random":
            print("[Warning] custom_transitions 为空，回退到全部随机")
            custom_pool = ALL_XFADE_TRANSITIONS

        if trans_d == 0:
            concat_inputs = "".join(f"[{i}:v]" for i in range(batch_size))
            video_filter_str = f"{concat_inputs}concat=n={batch_size}:v=1:a=0[vtout];"
            prev_label = "vtout"
        else:
            video_filters = []
            prev_label = "0:v"
            cumulative_frames = frame_counts[0]

            for i in range(batch_size - 1):
                inA = f"[{prev_label}]"
                inB = f"[{i+1}:v]"
                offset_frames = cumulative_frames - trans_d
                offset_sec = offset_frames / fps
                if offset_sec < 0:
                    offset_sec = 0.0

                # 转场选择逻辑
                if transition_type == "random":
                    actual_transition = random.choice(real_transitions)
                    print(f"[FFmpegImageSlideShow] 片段 {i}->{i+1} 转场: {actual_transition}")
                elif transition_type == "custom_random":
                    actual_transition = random.choice(custom_pool)
                    print(f"[FFmpegImageSlideShow] 片段 {i}->{i+1} 转场(自定义池): {actual_transition}")
                else:
                    actual_transition = transition_type

                out_label = f"vt{i+1}"
                video_filters.append(
                    f"{inA}{inB}xfade=transition={actual_transition}"
                    f":duration={transition_dur_sec:.6f}"
                    f":offset={offset_sec:.6f}[{out_label}];"
                )
                prev_label = out_label
                cumulative_frames += frame_counts[i+1] - trans_d

            video_filter_str = "".join(video_filters)

        filter_complex = (video_filter_str + audio_filter_str).rstrip(';')

        # ==================== 6. 确定最终输出路径（使用文件夹名称） ====================
        if output_folder and os.path.isdir(output_folder):
            out_dir = output_folder
        else:
            out_dir = folder_paths.get_temp_directory() if hasattr(folder_paths, 'get_temp_directory') else os.path.join(os.path.expanduser("~"), "ComfyUI", "temp")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            print(f"[FFmpegImageSlideShow] output_folder 无效，回退到临时目录: {out_dir}")

        # 使用文件夹名称作为文件名（清理非法字符）
        folder_name = os.path.basename(os.path.normpath(image_folder))
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', folder_name)
        if not safe_name:
            safe_name = "slideshow"
        output_filename = f"{safe_name}.mp4"
        output_path = os.path.join(out_dir, output_filename)
        os.makedirs(out_dir, exist_ok=True)

        # ==================== 7. 最终编码输出 ====================
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{prev_label}]",
        ]
        if use_audio:
            cmd += ["-map", "[aout]"]

        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
        ]
        if use_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        else:
            cmd += ["-an"]
        cmd += [output_path]

        run_ffmpeg(cmd, "Concatenate clips with audio")

        # ==================== 8. 清理临时文件 ====================
        for p in clip_paths:
            try:
                os.remove(p)
            except Exception as e:
                print(f"[Warning] 删除临时视频片段失败：{p} - {e}")
        
        for p in processed_paths:
            if p.startswith(os.path.abspath(temp_img_dir)):
                try:
                    os.remove(p)
                except Exception:
                    pass
        try:
            os.rmdir(temp_img_dir)
        except Exception:
            pass

        print(f"[FFmpegImageSlideShow] 成片已保存至: {output_path}")
        return {"ui": {"video": [output_path]}, "result": (output_path,)}
    
class FFmpegVideoSplitterZV:
    """
    使用 ffmpeg 按时间或帧数拆分视频，输出片段路径列表。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "输入视频的完整路径"
                }),
                "split_mode": (["time", "frames"], {
                    "default": "time",
                    "tooltip": "拆分模式：按时长(秒) 或 按帧数"
                }),
                "segment_time": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "step": 0.1,
                    "tooltip": "每段时长（秒），仅 split_mode = time 时有效"
                }),
                "segment_frames": ("INT", {
                    "default": 150,
                    "min": 1,
                    "step": 1,
                    "tooltip": "每段帧数，仅 split_mode = frames 时有效"
                }),
                "accurate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用精确切割（重编码），否则使用流复制（快速，切割点对齐关键帧）"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "输出目录（留空则使用视频所在目录下的 segments 文件夹）"
                }),
                "prefix": ("STRING", {
                    "default": "segment",
                    "tooltip": "输出文件名前缀"
                }),
                "ffmpeg_path": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "ffmpeg 可执行文件路径或命令名"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("split_video_paths", "count")
    FUNCTION = "split_video"
    CATEGORY = "ZVNodes/ffmpeg"
    OUTPUT_NODE = False

    def split_video(self, video, split_mode, segment_time, segment_frames, accurate,
                    output_dir, prefix, ffmpeg_path):
        # 1. 校验输入文件
        if not os.path.isfile(video):
            raise FileNotFoundError(f"视频文件不存在: {video}")

        # 2. 准备输出目录
        if output_dir.strip() == "":
            output_dir = os.path.join(os.path.dirname(video), "segments")
        os.makedirs(output_dir, exist_ok=True)

        # 3. 清理旧的同名前缀文件，避免混淆
        pattern = os.path.join(output_dir, f"{prefix}_*.mp4")
        for old_file in glob.glob(pattern):
            try:
                os.remove(old_file)
            except Exception:
                pass

        # 4. 构建输出文件名模板（ffmpeg segment 需要类似 "prefix_%03d.mp4"）
        output_template = os.path.join(output_dir, f"{prefix}_%03d.mp4")

        # 5. 构建 ffmpeg 命令
        cmd = [ffmpeg_path, "-i", video]

        # 如果是精确切割，需要重编码（可指定编码器，这里用 libx264 和 aac）
        if accurate:
            cmd += ["-c:v", "libx264", "-c:a", "aac", "-strict", "experimental"]
            # 重编码模式下，可加入 -force_key_frames 来精确定义关键帧位置，
            # 但 segment muxer 本身会对齐到关键帧，我们改为使用 -segment_time_delta 等。
            # 简单起见，精确模式直接重编码，segment 仍按时间/帧切割。
        else:
            cmd += ["-c", "copy"]  # 流复制，快速

        # 根据模式设置 segment 参数
        if split_mode == "time":
            cmd += ["-f", "segment", "-segment_time", str(segment_time)]
        else:  # frames
            cmd += ["-f", "segment", "-segment_frames", str(segment_frames)]

        # 通用 segment 参数：重置时间戳使每个片段从0开始，避免黑屏
        cmd += ["-reset_timestamps", "1"]

        # 如果精确模式，可以添加 -force_key_frames 使切割更精准（需编码支持）
        if accurate and split_mode == "time":
            # 在 segment_time 的整数倍位置强制关键帧
            cmd += ["-force_key_frames",
                    f"expr:gte(t,n_forced*{segment_time})"]
        elif accurate and split_mode == "frames":
            # 按帧强制关键帧，可通过 select 滤镜，这里简化
            pass

        # 输出模板
        cmd += [output_template]

        # 6. 执行命令
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 执行失败:\n{e.stderr}")

        # 7. 收集生成的片段文件，按数字编号排序
        segment_files = sorted(
            glob.glob(os.path.join(output_dir, f"{prefix}_*.mp4")),
            key=lambda x: int(re.search(rf"{prefix}_(\d+)\.mp4", os.path.basename(x)).group(1))
        )

        if not segment_files:
            raise RuntimeError("未生成任何视频片段，请检查 ffmpeg 输出。")

        # 8. 返回路径列表（换行分隔）和数量
        paths_str = "\n".join(segment_files)
        return (paths_str, len(segment_files))

class FFmpegVideoMergerZV:
    """
    使用 ffmpeg 将多个视频按顺序合并为一个视频。
    输入：多行字符串，每行一个视频路径
    输出：合并后的视频路径
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_paths": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "每行输入一个视频文件的完整路径"
                }),
                "reencode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用重编码（确保不同格式视频兼容）；关闭则使用流复制（快速，但要求编码一致）"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "输出目录（留空则使用第一个视频所在目录）"
                }),
                "output_filename": ("STRING", {
                    "default": "merged.mp4",
                    "tooltip": "输出文件名（含扩展名）"
                }),
                "ffmpeg_path": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "ffmpeg 可执行文件路径或命令"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_video_path",)
    FUNCTION = "merge_videos"
    CATEGORY = "video/ffmpeg"
    OUTPUT_NODE = False

    def merge_videos(self, video_paths, reencode, output_path, output_filename, ffmpeg_path):
        # 1. 解析输入路径，过滤空行
        paths = [line.strip() for line in video_paths.splitlines() if line.strip()]
        if not paths:
            raise ValueError("至少需要一个视频文件路径")

        # 2. 验证所有文件存在
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"视频文件不存在: {p}")

        # 3. 确定输出目录和文件
        first_video_dir = os.path.dirname(os.path.abspath(paths[0]))
        if output_path.strip() == "":
            output_dir = first_video_dir
        else:
            output_dir = output_path.strip()
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, output_filename)

        # 4. 生成 concat 文件列表（临时文件）
        # 对每个路径进行安全转义：将单引号替换为 '\''，然后包裹单引号
        def escape_path(p):
            # 把路径中的单引号替换为 '\''
            escaped = p.replace("'", "'\\''")
            return f"file '{escaped}'"

        concat_content = "\n".join(escape_path(p) for p in paths)

        # 写入临时文件（放在输出目录，避免跨盘权限问题）
        list_file = os.path.join(output_dir, "ffmpeg_concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            f.write(concat_content)

        # 5. 构建 ffmpeg 命令
        cmd = [
            ffmpeg_path,
            "-f", "concat",
            "-safe", "0",      # 允许使用绝对路径
            "-i", list_file,
        ]

        if reencode:
            # 重编码（可自定义编码器，这里用通用设置）
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k"]
        else:
            # 流复制
            cmd += ["-c", "copy"]

        cmd += ["-y", output_file]  # 自动覆盖已有文件

        # 6. 执行 ffmpeg
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, check=True)
        except subprocess.CalledProcessError as e:
            # 清理临时文件
            if os.path.exists(list_file):
                os.remove(list_file)
            raise RuntimeError(f"ffmpeg 合并失败:\n{e.stderr}")
        finally:
            # 删除临时文件（无论成功失败，如果还存在）
            if os.path.exists(list_file):
                try:
                    os.remove(list_file)
                except Exception:
                    pass

        # 7. 返回输出路径
        return (output_file,)
    
NODE_CONFIG = {
    "VideoCounterNodeZV": {"class": VideoCounterNodeZV, "name": "Count Video (Directory)"},
    "SaveVideoToPathZV": {"class": SaveVideoToPathZV, "name": "Save Video (Directory)"},
    "VideoSpeedZV": {"class": VideoSpeedZV, "name": "Video Speed"},
    "VideoSceneDetectorZV": {"class": VideoSceneDetectorZV, "name": "Video Scene Detector"},
    "LoadVideoFromDirZV":{"class": LoadVideoFromDirZV, "name": "Load One Video (Directory)"},
    "FFmpegImageSlideShowZV": {"class": FFmpegImageSlideShowZV, "name": "Image Slide Show (FFmpeg)"},
    "FFmpegVideoSplitterZV":{"class": FFmpegVideoSplitterZV, "name": "FFmpeg Video Splitter (Time/Frames)"},
    "FFmpegVideoMergerZV": {"class": FFmpegVideoMergerZV, "name": "FFmpeg Video Merger (Concat)"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)