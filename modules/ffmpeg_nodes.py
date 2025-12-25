import os
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image
import json
import folder_paths

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
    "VideoGeneratorFFmpegZV": VideoGeneratorFFmpegZV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGeneratorFFmpegZV": "FFmpeg视频生成器ZV"
}