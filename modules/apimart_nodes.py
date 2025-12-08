# apimart_nodes.py - 重构版
import os
import time
from typing import Dict, Any, Tuple, List, Optional
import torch

from .base_nodes import BaseAPIMartNode
from .utils import (
    pil2tensor, VideoAdapter, 
    batch_upload_image_to_apimart_cdn, 
    upload_video_to_apimart_cdn, 
    image_to_temp_png,
    image_to_temp_file
)

# 导入文件夹路径工具
try:
    import folder_paths
except ImportError:
    class folder_paths:
        @staticmethod
        def get_output_directory():
            return os.path.join(os.getcwd(), "output")


class ApimartText2VideoSubmitZV(BaseAPIMartNode):
    """文生视频提交任务类"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "10", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        try:
            self.initialize(api_key)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": int(duration),
            }
            
            code, body = self.client.submit_video_generation(payload)
            task_id = self.parser.extract_task_id(body)
            
            report = f"HTTP {code} - {'提交成功' if code == 200 else '提交失败'}"
            if task_id:
                self.task_manager.add_task(task_id, "text2video")
                report += f" | task_id: {task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                report += f" | 未返回task_id"
                if error_msg:
                    report += f" | 错误: {error_msg}"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交文生视频任务失败")


class ApimartImage2VideoSubmitZV(BaseAPIMartNode):
    """图生视频提交任务类（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "10", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, image, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        try:
            self.initialize(api_key)
            
            # 使用基类的通用方法
            body, method, task_id, error = self._handle_image_upload_and_submit(
                image=image,
                prompt=prompt,
                api_key=api_key,
                model=model,
                aspect_ratio=aspect_ratio,
                duration=int(duration),
                task_type="image2video"
            )
            
            # 处理错误情况
            if error:
                return (error, "")
            
            # 生成报告
            code = body.get("status_code", 0) if isinstance(body, dict) else 0
            if code == 0:
                code = 200 if task_id else 500  # 如果没有状态码，根据是否有task_id推断
            
            report = f"HTTP {code} - {'提交成功' if task_id else '提交失败'} | {method}"
            
            if task_id:
                report += f" | task_id: {task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                if error_msg:
                    report += f" | 错误: {error_msg}"
                else:
                    report += " | 未返回task_id"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交图生视频任务失败")


class Veo31Image2VideoSubmitZV(BaseAPIMartNode):
    """Veo3.1图生视频提交任务类（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "veo3.1-fast", "choices": ["veo3.1-fast", "veo3.1-quality"]}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, image, prompt: str, api_key: str, model: str):
        try:
            self.initialize(api_key)
            
            # Veo3.1的固定参数
            aspect_ratio = "16:9"
            duration = 8
            
            # 使用基类的通用方法
            body, method, task_id, error = self._handle_image_upload_and_submit(
                image=image,
                prompt=prompt,
                api_key=api_key,
                model=model,
                aspect_ratio=aspect_ratio,
                duration=duration,
                task_type="veo31_image2video"
            )
            
            # 处理错误情况
            if error:
                return (error, "")
            
            # 生成报告
            code = body.get("status_code", 0) if isinstance(body, dict) else 0
            if code == 0:
                code = 200 if task_id else 500
            
            # 生成详细报告
            try:
                receipt = str(body)[:200] + "..." if len(str(body)) > 200 else str(body)
            except:
                receipt = str(body)
            
            report = f"HTTP {code} | 模型: {model} | AR: {aspect_ratio} | 时长: {duration}s | {method} | 回执: {receipt}"
            
            if task_id:
                report += f" | task_id: {task_id} 已保存"
            else:
                report += " | 未返回task_id"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交Veo3.1图生视频任务失败")


class ApimartDownloadSavedTaskVideoZV(BaseAPIMartNode):
    """下载已保存任务视频类（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 12, "min": 1, "max": 60}),
                "retry_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "report")
    FUNCTION = "process"
    
    def process(self, api_key: str, task_id: str = "", max_retries: int = 12, 
                retry_interval: int = 5, download_with_progress: bool = True):
        try:
            self.initialize(api_key)
            
            # 获取任务ID
            manual_id = (task_id or "").strip()
            if manual_id:
                selected_task_id = manual_id
            else:
                task_record = self.task_manager.get_first_task()
                if not task_record:
                    return (VideoAdapter(None), "", "没有已保存的任务ID")
                selected_task_id = task_record.task_id
            
            # 使用增强的轮询方法
            def progress_callback(progress, status):
                print(f"任务进度: {progress}%, 状态: {status}")
            
            video_url, status, retry_count, progress, full_response = self._poll_task_status_with_progress(
                task_id=selected_task_id,
                api_key=api_key,
                max_retries=max_retries,
                retry_interval=retry_interval,
                progress_callback=progress_callback
            )
            
            if not video_url:
                return (VideoAdapter(None), "", f"任务未完成或无视频链接 | status={status}, progress={progress}%")
            
            # 下载视频
            import folder_paths
            base = folder_paths.get_output_directory()
            output_name = f"apimart_{selected_task_id}.mp4"
            output_path = os.path.join(base, output_name)
            
            if download_with_progress:
                # 使用带进度的下载
                def download_progress(percent, msg):
                    print(f"下载: {msg}")
                
                success, msg, file_size = self._download_media_with_progress(
                    url=video_url,
                    output_path=output_path,
                    task_id=selected_task_id,
                    progress_callback=download_progress
                )
            else:
                # 普通下载
                success, msg = self._download_with_retries(video_url, output_path)
                file_size = None
            
            if not success:
                return (VideoAdapter(None), "", f"下载失败: {msg}")
            
            # 创建视频适配器
            adapter = VideoAdapter(output_path)
            
            # 生成详细报告
            report_parts = [
                f"任务ID: {selected_task_id}",
                f"状态: {status}",
                f"最终进度: {progress or 100}%",
                f"重试次数: {retry_count}",
            ]
            
            if file_size:
                report_parts.append(f"文件大小: {file_size}字节")
            
            if output_path:
                report_parts.append(f"保存路径: {output_path}")
            
            report = " | ".join(report_parts)
            
            # 从任务管理器中移除任务
            self.task_manager.remove_task(selected_task_id)
            
            return (adapter, output_path, report)
            
        except Exception as e:
            return self.handle_error(e, "下载视频任务失败")


class ApimartRemixVideoSubmitZV(BaseAPIMartNode):
    """视频Remix提交任务类"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "duration": ("STRING", {"default": "15", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            },
            "optional": {
                "video_id": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("task_id",)
    FUNCTION = "process"
    
    def process(self, video, prompt: str, api_key: str, duration: str, model: str, video_id: str = ""):
        try:
            self.initialize(api_key)
            
            final_duration = int(duration)
            vid = (video_id or "").strip()
            task_id = None
            
            # 优先使用video_id
            if vid:
                # 使用video_id进行remix
                url = f"https://api.apimart.ai/v1/videos/{vid}/remix"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "duration": final_duration,
                    "aspect_ratio": "16:9",
                }
                
                code, body = self.client.request_with_retry(
                    "POST",
                    url,
                    json=payload,
                    headers=self.client.config.get_headers(self.client.api_key)
                )
                
                task_id = self.parser.extract_task_id(body)
                
            else:
                # 上传视频到CDN然后进行remix
                cdn_url = upload_video_to_apimart_cdn(video, api_key)
                if not cdn_url:
                    return ("视频上传到CDN失败",)
                
                # 尝试不同的字段名
                base_url = "https://api.apimart.ai/v1/videos/remix"
                for field in ["url", "video_url", "video_urls"]:
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "duration": final_duration,
                    }
                    
                    if field == "video_urls":
                        payload[field] = [cdn_url]
                    else:
                        payload[field] = cdn_url
                    
                    code, body = self.client.request_with_retry(
                        "POST",
                        base_url,
                        json=payload,
                        headers=self.client.config.get_headers(self.client.api_key)
                    )
                    
                    if code == 200:
                        task_id = self.parser.extract_task_id(body)
                        break
            
            if task_id:
                self.task_manager.add_task(task_id, "video_remix")
            
            return (task_id or "",)
            
        except Exception as e:
            return self.handle_error(e, "提交视频Remix任务失败")


class ApimartRemixByTaskIdSubmitZV(BaseAPIMartNode):
    """通过TaskID视频Remix提交类"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "15", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, task_id: str, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        try:
            self.initialize(api_key)
            
            vid = (task_id or "").strip()
            if not vid:
                return ("未提供有效的task_id", "")
            
            final_duration = int(duration)
            
            # 使用video_id进行remix
            url = f"https://api.apimart.ai/v1/videos/{vid}/remix"
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": final_duration,
                "aspect_ratio": aspect_ratio,
            }
            
            code, body = self.client.request_with_retry(
                "POST",
                url,
                json=payload,
                headers=self.client.config.get_headers(self.client.api_key)
            )
            
            new_task_id = self.parser.extract_task_id(body)
            
            report = f"HTTP {code} - {'提交成功' if code == 200 else '提交失败'}"
            if new_task_id:
                self.task_manager.add_task(new_task_id, "remix_by_taskid")
                report += f" | task_id: {new_task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                if error_msg:
                    report += f" | 错误: {error_msg}"
                else:
                    report += " | 未返回task_id"
            
            return (report, new_task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "通过TaskID提交视频Remix失败")


class ApimartSeedream40ImageSubmitZV(BaseAPIMartNode):
    """Seedream 4.0 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "可爱的熊猫在竹林中玩耍"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]}),
                "n": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "model": ("STRING", {"default": "doubao-seedance-4-0", "choices": ["doubao-seedance-4-0"]}),
                "optimize_prompt_options": ("STRING", {"default": "standard", "choices": ["standard", "fast"]}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "sequential_image_generation": ("STRING", {"default": "disabled", "choices": ["disabled", "auto"]}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, prompt: str, api_key: str, size: str, n: int, model: str,
               optimize_prompt_options: str, watermark: bool,
               image=None, sequential_image_generation: str = "disabled", max_images: int = 3):
        try:
            self.initialize(api_key)
            
            # 构建基础payload
            payload = {
                "model": model,
                "prompt": prompt[:1000],  # 限制1000字符
                "size": size,
                "n": n,
                "optimize_prompt_options": optimize_prompt_options,
                "watermark": watermark,
            }
            
            # 处理参考图像
            if image is not None:
                cdn_urls = batch_upload_image_to_apimart_cdn(image, api_key, max_side=2048)
                if cdn_urls:
                    payload["image_urls"] = cdn_urls
            
            # 处理组图生成
            if sequential_image_generation == "auto":
                payload["sequential_image_generation"] = "auto"
                payload["sequential_image_generation_options"] = {
                    "max_images": max_images
                }
            
            # 提交图像生成任务
            code, body = self.client.submit_image_generation(payload)
            task_id = self.parser.extract_task_id(body)
            
            report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 数量: {n}"
            if task_id:
                self.task_manager.add_task(task_id, "seedream40_image")
                report += f" | task_id: {task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                if error_msg:
                    report += f" | 错误: {error_msg}"
                else:
                    report += " | 未返回 task_id"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交Seedream 4.0图像生成任务失败")


class ApimartSeedream45ImageSubmitZV(BaseAPIMartNode):
    """Seedream 4.5 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "可爱的熊猫在竹林中玩耍"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9", "2k", "4k"]}),
                "n": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "model": ("STRING", {"default": "doubao-seedance-4-5", "choices": ["doubao-seedance-4-5"]}),
                "optimize_prompt_options_mode": ("STRING", {"default": "standard", "choices": ["standard", "fast"]}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "sequential_image_generation": ("STRING", {"default": "disabled", "choices": ["disabled", "auto"]}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, prompt: str, api_key: str, size: str, n: int, model: str,
               optimize_prompt_options_mode: str, watermark: bool,
               image=None, sequential_image_generation: str = "disabled", max_images: int = 3):
        try:
            self.initialize(api_key)
            
            # 构建基础payload
            payload = {
                "model": model,
                "prompt": prompt,  # 4.5没有字符限制
                "size": size,
                "n": n,
                "optimize_prompt_options": {
                    "mode": optimize_prompt_options_mode
                },
                "watermark": watermark,
            }
            
            # 处理参考图像
            if image is not None:
                cdn_urls = batch_upload_image_to_apimart_cdn(image, api_key, max_side=2048)
                if cdn_urls:
                    payload["image_urls"] = cdn_urls
            
            # 处理组图生成
            if sequential_image_generation == "auto":
                payload["sequential_image_generation"] = "auto"
                payload["sequential_image_generation_options"] = {
                    "max_images": max_images
                }
            
            # 提交图像生成任务
            code, body = self.client.submit_image_generation(payload)
            task_id = self.parser.extract_task_id(body)
            
            report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 数量: {n}"
            if task_id:
                self.task_manager.add_task(task_id, "seedream45_image")
                report += f" | task_id: {task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                if error_msg:
                    report += f" | 错误: {error_msg}"
                else:
                    report += " | 未返回 task_id"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交Seedream 4.5图像生成任务失败")


class ApimartNanoBananaProImageSubmitZV(BaseAPIMartNode):
    """NanoBananaPro (Gemini-3-Pro-Image-preview) 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "月光下的竹林小径"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]}),
                "resolution": ("STRING", {"default": "1K", "choices": ["1K", "2K", "4K"]}),
                "model": ("STRING", {"default": "gemini-3-pro-image-preview", "choices": ["gemini-3-pro-image-preview"]}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "mask": ("IMAGE", {}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    FUNCTION = "process"
    
    def process(self, prompt: str, api_key: str, size: str, resolution: str, model: str,
               image=None, mask=None):
        try:
            self.initialize(api_key)
            
            # 构建基础payload - NanoBananaPro固定n=1
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": 1,  # 固定为1
                "resolution": resolution,
            }
            
            # 处理参考图像
            if image is not None:
                cdn_urls = batch_upload_image_to_apimart_cdn(image, api_key, max_side=2048)
                if cdn_urls:
                    payload["image_urls"] = cdn_urls
            
            # 处理蒙版图像
            if mask is not None:
                # 蒙版图像必须是PNG格式
                mask_temp_file = image_to_temp_file(mask, max_side=1024, format="PNG")
                if mask_temp_file:
                    try:
                        # 使用工具函数上传蒙版
                        from .utils import CDNUploader
                        uploader = CDNUploader(api_key)
                        mask_url = uploader.upload_image(mask, max_side=1024)
                        if mask_url:
                            payload["mask_url"] = mask_url
                    finally:
                        # 清理临时文件
                        if mask_temp_file and os.path.exists(mask_temp_file):
                            os.unlink(mask_temp_file)
            
            # 提交图像生成任务
            code, body = self.client.submit_image_generation(payload)
            task_id = self.parser.extract_task_id(body)
            
            report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 分辨率: {resolution}"
            if task_id:
                self.task_manager.add_task(task_id, "nanobananapro_image")
                report += f" | task_id: {task_id} 已保存"
            else:
                error_msg = self.parser.extract_error_message(body)
                if error_msg:
                    report += f" | 错误: {error_msg}"
                else:
                    report += " | 未返回 task_id"
            
            return (report, task_id or "")
            
        except Exception as e:
            return self.handle_error(e, "提交NanoBananaPro图像生成任务失败")


class ApimartDownloadSavedTaskImageZV(BaseAPIMartNode):
    """下载已保存的图像生成任务结果（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 12, "min": 1, "max": 60}),
                "retry_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "process"
    
    def process(self, api_key: str, task_id: str = "", max_retries: int = 12, retry_interval: int = 5):
        try:
            self.initialize(api_key)
            
            # 获取任务ID
            manual_id = (task_id or "").strip()
            if manual_id:
                selected_task_id = manual_id
            else:
                task_record = self.task_manager.get_first_task()
                if not task_record:
                    # 返回空批次
                    empty_batch = torch.zeros((0, 3, 512, 512)) if torch else None
                    return (empty_batch, "没有已保存的任务ID")
                selected_task_id = task_record.task_id
            
            # 使用基类的轮询方法获取图像URLs
            image_urls, status, retry_count = self._poll_task_status(
                task_id=selected_task_id,
                max_retries=max_retries,
                retry_interval=retry_interval,
                extract_func=self.parser.extract_image_urls
            )
            
            # 检查是否获取到图像URLs
            if not image_urls:
                empty_batch = torch.zeros((0, 3, 512, 512)) if torch else None
                return (empty_batch, f"任务未完成或无图像链接 | status={status} | task_id={selected_task_id}")
            
            # 下载并处理所有图像
            downloaded_images = []
            download_dir = os.path.join(folder_paths.get_output_directory(), "apimart_images")
            os.makedirs(download_dir, exist_ok=True)
            
            success_count = 0
            failed_urls = []
            
            for i, url in enumerate(image_urls):
                try:
                    # 生成文件名
                    filename = f"apimart_{selected_task_id}_{i+1}.png"
                    filepath = os.path.join(download_dir, filename)
                    
                    # 下载图像
                    if self._download_image(url, filepath):
                        # 加载图像为tensor
                        tensor = self._load_image_to_tensor(filepath)
                        if tensor is not None:
                            downloaded_images.append(tensor)
                            success_count += 1
                        else:
                            failed_urls.append(f"图像{i+1}加载失败")
                    else:
                        failed_urls.append(f"图像{i+1}下载失败")
                        
                except Exception as e:
                    failed_urls.append(f"图像{i+1}处理失败: {str(e)}")
            
            # 组合所有图像为一个批次
            if downloaded_images:
                try:
                    batch = torch.cat(downloaded_images, dim=0)
                except Exception as e:
                    batch = torch.zeros((0, 3, 512, 512)) if torch else None
                    return (batch, f"批次组合失败: {str(e)} | task_id={selected_task_id}")
            else:
                batch = torch.zeros((0, 3, 512, 512)) if torch else None
            
            # 生成报告
            report_parts = []
            if retry_count > 0:
                report_parts.append(f"重试 {retry_count} 次后成功")
            report_parts.append(f"下载完成 | task_id={selected_task_id}")
            report_parts.append(f"成功: {success_count}/{len(image_urls)}")
            
            if failed_urls:
                report_parts.append(f"失败: {', '.join(failed_urls[:3])}")
                if len(failed_urls) > 3:
                    report_parts.append(f"...等{len(failed_urls)}个失败")
            
            report = " | ".join(report_parts)
            
            # 下载成功后删除该任务
            if success_count > 0:
                self.task_manager.remove_task(selected_task_id)
            
            return (batch, report)
            
        except Exception as e:
            return self.handle_error(e, "下载图像任务失败")
    
    def _download_image(self, url: str, path: str) -> bool:
        """下载单个图像文件"""
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
            
        except Exception:
            return False
    
    def _load_image_to_tensor(self, image_path: str):
        """将图像文件加载为torch.Tensor"""
        try:
            from PIL import Image
            
            # 打开图像
            pil_img = Image.open(image_path)
            
            # 确保图像是RGB格式
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            # 转换为tensor
            return pil2tensor(pil_img)
            
        except Exception:
            return None


# ==================== 节点注册 ====================

NODE_CLASS_MAPPINGS = {
    "ApimartText2VideoSubmitZV": ApimartText2VideoSubmitZV,
    "ApimartImage2VideoSubmitZV": ApimartImage2VideoSubmitZV,
    "ApimartDownloadSavedTaskVideoZV": ApimartDownloadSavedTaskVideoZV,
    "ApimartRemixVideoSubmitZV": ApimartRemixVideoSubmitZV,
    "ApimartRemixByTaskIdSubmitZV": ApimartRemixByTaskIdSubmitZV,
    "Veo31Image2VideoSubmitZV": Veo31Image2VideoSubmitZV,
    "ApimartSeedream40ImageSubmitZV": ApimartSeedream40ImageSubmitZV,
    "ApimartSeedream45ImageSubmitZV": ApimartSeedream45ImageSubmitZV,
    "ApimartNanoBananaProImageSubmitZV": ApimartNanoBananaProImageSubmitZV,
    "ApimartDownloadSavedTaskImageZV": ApimartDownloadSavedTaskImageZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApimartText2VideoSubmitZV": "文生视频提交任务(apimart)",
    "ApimartImage2VideoSubmitZV": "图生视频提交任务(apimart)",
    "ApimartDownloadSavedTaskVideoZV": "下载已保存任务视频(apimart)",
    "ApimartRemixVideoSubmitZV": "视频 Remix 提交任务(apimart)",
    "ApimartRemixByTaskIdSubmitZV": "通过 TaskID 视频 Remix 提交(apimart)",
    "Veo31Image2VideoSubmitZV": "veo3.1 图生视频提交任务(apimart)",
    "ApimartSeedream40ImageSubmitZV": "Seedream 4.0 图像生成(apimart)",
    "ApimartSeedream45ImageSubmitZV": "Seedream 4.5 图像生成(apimart)",
    "ApimartNanoBananaProImageSubmitZV": "NanoBananaPro 图像生成(apimart)",
    "ApimartDownloadSavedTaskImageZV": "下载已保存任务图像(apimart)",
}