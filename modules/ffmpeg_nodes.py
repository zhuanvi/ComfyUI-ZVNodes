import os
import re
import glob
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image
import json
import folder_paths

class FFmpegVideoSplitterZV:
    """
    使用 ffmpeg 按时间或帧数拆分视频，输出片段路径列表。
    """
    def __init__(self):
        pass

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
    CATEGORY = "video/ffmpeg"
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

class VideoGeneratorFFmpegZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),  # 假设有音频输入节点
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "slider"
                }),
                "output_format": (["mp4", "avi", "mov", "mkv", "webm"], {"default": "mp4"}),
                "encoder": (["libx264", "libx265", "libvpx-vp9", "h264_nvenc", "hevc_nvenc"], {"default": "libx264"}),
                "quality": (["low", "medium", "high", "lossless"], {"default": "medium"}),
                "output_path": ("STRING", {
                    "default": "./output/video.mp4",
                    "multiline": False
                }),
            },
            "optional": {
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "audio_sync": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "ZVNodes/ffmpeg"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = tempfile.gettempdir()

    def images_to_temp_frames(self, images):
        """将图像批次保存为临时帧"""
        temp_dir = tempfile.mkdtemp(prefix="comfy_video_")
        
        for i, img_tensor in enumerate(images):
            # 转换为numpy数组
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 转换为PIL图像
            if img_np.shape[-1] == 3:  # RGB
                img = Image.fromarray(img_np, 'RGB')
            elif img_np.shape[-1] == 4:  # RGBA
                img = Image.fromarray(img_np, 'RGBA')
            else:
                img = Image.fromarray(img_np.squeeze(), 'L')
            
            # 保存为PNG序列
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            img.save(frame_path)
        
        return temp_dir, len(images)

    def get_encoder_settings(self, encoder, quality, format):
        """获取编码器设置"""
        settings = {
            "libx264": {
                "codec": "libx264",
                "crf": {"low": 28, "medium": 23, "high": 18, "lossless": 0}[quality],
                "preset": "medium",
                "pix_fmt": "yuv420p"
            },
            "libx265": {
                "codec": "libx265",
                "crf": {"low": 32, "medium": 28, "high": 23, "lossless": 0}[quality],
                "preset": "medium",
                "pix_fmt": "yuv420p"
            },
            "libvpx-vp9": {
                "codec": "libvpx-vp9",
                "crf": {"low": 40, "medium": 32, "high": 24, "lossless": 0}[quality],
                "cpu-used": 2,
                "row-mt": 1,
                "pix_fmt": "yuv420p"
            },
            "h264_nvenc": {
                "codec": "h264_nvenc",
                "cq": {"low": 32, "medium": 26, "high": 20, "lossless": 0}[quality],
                "preset": "p4",
                "pix_fmt": "yuv420p"
            },
            "hevc_nvenc": {
                "codec": "hevc_nvenc",
                "cq": {"low": 32, "medium": 26, "high": 20, "lossless": 0}[quality],
                "preset": "p4",
                "pix_fmt": "yuv420p"
            }
        }
        
        return settings.get(encoder, settings["libx264"])

    def generate_video(self, images, audio, fps, output_format, encoder, quality, 
                      output_path, audio_volume=1.0, audio_sync=0.0):
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存图像为临时帧
        frames_dir, num_frames = self.images_to_temp_frames(images)
        
        try:
            # 获取编码器设置
            encoder_settings = self.get_encoder_settings(encoder, quality, output_format)
            
            # 构建FFmpeg命令
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%06d.png"),
            ]
            
            # 添加音频输入
            if audio and os.path.exists(audio):
                cmd.extend([
                    "-i", audio,
                    "-af", f"volume={audio_volume}"
                ])
                
                # 视频编码参数
                cmd.extend([
                    "-c:v", encoder_settings["codec"],
                    "-c:a", "aac",  # 音频编码为AAC
                    "-b:a", "192k",  # 音频比特率
                ])
                
                # 添加编码器特定参数
                if "crf" in encoder_settings:
                    cmd.extend(["-crf", str(encoder_settings["crf"])])
                elif "cq" in encoder_settings:
                    cmd.extend(["-cq", str(encoder_settings["cq"])])
                
                # 音频同步
                if abs(audio_sync) > 0.001:
                    cmd.extend(["-af", f"adelay={int(audio_sync*1000)}|{int(audio_sync*1000)}"])
                
                # 使用较短的一方
                cmd.extend(["-shortest"])
            else:
                # 无音频版本
                cmd.extend([
                    "-c:v", encoder_settings["codec"],
                ])
                
                if "crf" in encoder_settings:
                    cmd.extend(["-crf", str(encoder_settings["crf"])])
                elif "cq" in encoder_settings:
                    cmd.extend(["-cq", str(encoder_settings["cq"])])
            
            # 添加预设和像素格式
            if "preset" in encoder_settings:
                cmd.extend(["-preset", encoder_settings["preset"]])
            
            if "pix_fmt" in encoder_settings:
                cmd.extend(["-pix_fmt", encoder_settings["pix_fmt"]])
            
            # 添加CPU优化参数
            if encoder == "libvpx-vp9" and "cpu-used" in encoder_settings:
                cmd.extend(["-cpu-used", str(encoder_settings["cpu-used"])])
                if encoder_settings.get("row-mt", 0):
                    cmd.extend(["-row-mt", "1"])
            
            # 添加硬件加速（如果有）
            if encoder in ["h264_nvenc", "hevc_nvenc"]:
                cmd.extend(["-rc", "vbr"])
            
            # 输出文件路径
            cmd.append(output_path)
            
            # 执行FFmpeg命令
            print(f"执行FFmpeg命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                print(f"FFmpeg错误: {result.stderr}")
                raise Exception(f"视频生成失败: {result.stderr}")
            
            print(f"视频生成成功: {output_path}")
            
            return (output_path,)
            
        finally:
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(frames_dir)
            except:
                pass

# 注册节点
NODE_CLASS_MAPPINGS = {
    "FFmpegVideoSplitterZV": FFmpegVideoSplitterZV,
    "VideoGeneratorFFmpegZV": VideoGeneratorFFmpegZV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FFmpegVideoSplitterZV": "FFmpeg Video Splitter (Time/Frames)",
    "VideoGeneratorFFmpegZV": "FFmpeg视频生成器ZV"
}

# 可选：将节点注册到 ComfyUI
NODE_CLASS_MAPPINGS = {
    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FFmpegVideoSplitter": "FFmpeg Video Splitter (Time/Frames)"
}