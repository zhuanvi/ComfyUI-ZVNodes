# grsai_nodes.py
import json
from typing import Dict, List, Optional, Tuple, Any
import torch

from .base_nodes import GrsaiBaseClient
from .utils import MediaProcessor, VideoAdapter, batch_upload_image_to_apimart_cdn
from comfy.comfy_types import IO


class GrsaiBaseNode:
    """Grsai节点基类"""
    
    def __init__(self):
        self.client = None
        self.media_processor = MediaProcessor()
    
    def initialize(self, api_key: str):
        """初始化客户端"""
        if not self.client:
            self.client = GrsaiBaseClient(api_key)
    
    def get_empty_response(self, return_types: Tuple) -> Tuple:
        """获取空响应"""
        empty_values = []
        for return_type in return_types:
            if return_type == "STRING":
                empty_values.append("")
            elif return_type == "IMAGE":
                empty_values.append(torch.zeros((0, 3, 512, 512)))
            elif return_type == "VIDEO":
                empty_values.append(VideoAdapter(None))
            elif return_type == "INT":
                empty_values.append(0)
            else:
                empty_values.append("")
        return tuple(empty_values)
    
    def process_reference_images(self, reference_images, api_key: str) -> List[str]:
        """处理参考图像"""
        if reference_images is None:
            return []
        
        # 尝试上传到CDN
        cdn_urls = batch_upload_image_to_apimart_cdn(reference_images, api_key, max_side=2048)
        if cdn_urls:
            return cdn_urls
        
        # 转换为Base64作为备选
        base64_images = self.client.image_to_base64(reference_images, include_prefix=True)
        if isinstance(base64_images, list):
            return base64_images
        elif base64_images:
            return [base64_images]
        
        return []


class GrsaiSoraVideoNodeZV(GrsaiBaseNode):
    """Sora视频生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["sora-2"], {"default": "sora-2"}),
                "aspect_ratio": (["9:16", "16:9", "1:1", "4:3", "3:2"], {"default": "9:16"}),
                "duration": ("INT", {"default": 10, "min": 10, "max": 15, "step": 5}),
                "size": (["small", "large"], {"default": "small"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "remix_target_id": ("STRING", {"default": "", "multiline": False}),
                "characters_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "task_id", "pid")
    FUNCTION = "generate_video"
    CATEGORY = "Sora Video"
    
    def generate_video(self, api_key, prompt, model, aspect_ratio, duration, size, 
                      use_cn_endpoint, response_mode, webhook_url, shut_progress,
                      reference_image=None, remix_target_id="", characters_json="[]"):
        
        self.initialize(api_key)
        
        # 准备请求体
        payload = {
            "model": model,
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "duration": duration,
            "size": size,
            "shutProgress": shut_progress
        }
        
        # 处理参考图像
        if reference_image is not None:
            image_urls = self.process_reference_images(reference_image, api_key)
            if image_urls:
                payload["url"] = image_urls[0]
        
        # 处理续作ID
        if remix_target_id:
            payload["remixTargetId"] = remix_target_id
            
        # 处理角色JSON
        try:
            characters = json.loads(characters_json)
            if characters:
                payload["characters"] = characters
        except json.JSONDecodeError:
            print(f"警告: 无效的角色JSON: {characters_json}")
        
        # 发送请求
        result, error = self.client.make_request(
            endpoint="/v1/video/sora-video",
            payload=payload,
            response_mode=response_mode,
            webhook_url=webhook_url,
            use_cn_endpoint=use_cn_endpoint,
            timeout=300
        )
        
        if error:
            print(f"生成视频错误: {error}")
            return self.get_empty_response(self.RETURN_TYPES)
        
        # 处理结果
        if response_mode == "polling":
            if result and result.get('code') == 0:
                task_id = result['data']['id']
                print(f"任务创建成功。任务ID: {task_id}")
                return ("", task_id, "")
        elif result and result.get('status') == 'succeeded':
            results = result.get('results', [])
            if results:
                video_url = results[0].get('url', '')
                pid = results[0].get('pid', '')
                task_id = result.get('id', '')
                print(f"视频生成成功! URL: {video_url}")
                return (video_url, task_id, pid)
        
        return self.get_empty_response(self.RETURN_TYPES)


class GrsaiNanoBananaNodeZV(GrsaiBaseNode):
    """Nano Banana图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["nano-banana-fast", "nano-banana", "nano-banana-pro"], {"default": "nano-banana-fast"}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"], 
                               {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "task_id", "content")
    FUNCTION = "generate_image"
    CATEGORY = "Nano Banana"
    
    def generate_image(self, api_key, prompt, model, aspect_ratio, image_size, 
                      use_cn_endpoint, response_mode, webhook_url, shut_progress,
                      reference_images=None):
        
        self.initialize(api_key)
        
        # 准备请求体
        payload = {
            "model": model,
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "imageSize": image_size,
            "shutProgress": shut_progress
        }
        
        # 处理参考图像
        if reference_images is not None:
            image_urls = self.process_reference_images(reference_images, api_key)
            if image_urls:
                payload["urls"] = image_urls
        
        # 发送请求
        result, error = self.client.make_request(
            endpoint="/v1/draw/nano-banana",
            payload=payload,
            response_mode=response_mode,
            webhook_url=webhook_url,
            use_cn_endpoint=use_cn_endpoint,
            timeout=300
        )
        
        if error:
            print(f"生成图像错误: {error}")
            return self.get_empty_response(self.RETURN_TYPES)
        
        # 处理结果
        if response_mode == "polling":
            if result and result.get('code') == 0:
                task_id = result['data']['id']
                print(f"任务创建成功。任务ID: {task_id}")
                return ("", task_id, "")
        elif result and result.get('status') == 'succeeded':
            results = result.get('results', [])
            if results:
                image_url = results[0].get('url', '')
                content = results[0].get('content', '')
                task_id = result.get('id', '')
                print(f"图像生成成功! URL: {image_url}")
                return (image_url, task_id, content)
        
        return self.get_empty_response(self.RETURN_TYPES)


class GrsaiResultNodeZV(GrsaiBaseNode):
    """统一结果获取节点 - 支持Sora视频和Nano Banana图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "max_retries": ("INT", {"default": 30, "min": 1, "max": 100}),
                "retry_delay": ("INT", {"default": 5, "min": 1, "max": 30}),
                "download_media": ("BOOLEAN", {"default": True}),
                "auto_detect_type": ("BOOLEAN", {"default": True}),
                "media_type": (["auto", "image", "video"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "IMAGE", IO.VIDEO)
    RETURN_NAMES = ("url", "task_id", "pid", "content", "progress", "status", "image", "video")
    FUNCTION = "get_result"
    CATEGORY = "Grsai API"
    OUTPUT_NODE = True
    
    def get_result(self, api_key, task_id, use_cn_endpoint, max_retries, retry_delay,
                  download_media, auto_detect_type, media_type):
        
        self.initialize(api_key)
        
        print(f"轮询任务结果，任务ID: {task_id}")
        
        # 轮询任务状态
        data, error, attempts = self.client.poll_task_result(
            task_id=task_id,
            use_cn_endpoint=use_cn_endpoint,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if error:
            print(f"轮询错误: {error}")
            return self._get_error_response(task_id)
        
        if not data:
            return self._get_error_response(task_id, "未获取到任务数据")
        
        status = data.get('status', '')
        progress = data.get('progress', 0)
        results = data.get('results', [])
        
        print(f"进度: {progress}%, 状态: {status}")
        
        if status == 'succeeded' and results:
            return self._process_successful_result(results[0], data, task_id, download_media, auto_detect_type, media_type)
        elif status == 'failed':
            error_msg = data.get('error', data.get('failure_reason', '未知错误'))
            print(f"任务失败: {error_msg}")
            return self._get_error_response(task_id, error_msg, progress, status)
        else:
            return self._get_error_response(task_id, f"任务状态异常: {status}", progress, status)
    
    def _process_successful_result(self, result_data, task_data, task_id, download_media, auto_detect_type, media_type):
        """处理成功的结果"""
        url = result_data.get('url', '')
        pid = result_data.get('pid', '')
        content = result_data.get('content', '')
        progress = task_data.get('progress', 0)
        status = task_data.get('status', '')
        
        print(f"任务完成成功! URL: {url}")
        
        # 初始化输出
        image_tensor = torch.zeros((0, 3, 512, 512))
        video_adapter = VideoAdapter(None)
        
        # 检测媒体类型
        detected_type = self.media_processor.detect_media_type(result_data, media_type, auto_detect_type)
        
        # 下载媒体文件
        if download_media and url:
            if detected_type == "image":
                image_tensor = self.media_processor.download_image(url) or image_tensor
            elif detected_type == "video":
                video_path, error = self.media_processor.download_video(url, task_id)
                if video_path:
                    video_adapter = VideoAdapter(video_path)
        
        return (url, task_id, pid, content, progress, status, image_tensor, video_adapter)
    
    def _get_error_response(self, task_id, error_msg="", progress=0, status="error"):
        """获取错误响应"""
        return ("" if error_msg else "", task_id, "", "", progress, status, 
                torch.zeros((0, 3, 512, 512)), VideoAdapter(None))


class GrsaiLoadImageFromPathNodeZV(GrsaiBaseNode):
    """从路径加载图像节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "Nano Banana"
    
    def load_image(self, image_path):
        image_tensor = self.media_processor.load_image_from_path(image_path)
        if image_tensor is not None:
            return (image_tensor,)
        return (torch.zeros((0, 3, 512, 512)),)


class GrsaiSoraUploadCharacterNodeZV(GrsaiBaseNode):
    """上传角色视频节点（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "character_video_url": ("STRING", {"default": "", "multiline": False}),
                "timestamps": ("STRING", {"default": "0,3", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "character_video": ("VIDEO",),  # 新增：直接上传视频文件
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "task_id")
    FUNCTION = "upload_character"
    CATEGORY = "Sora Video"
    
    def upload_character(self, api_key, character_video_url, timestamps, 
                        use_cn_endpoint, response_mode, webhook_url, shut_progress,
                        character_video=None):
        """
        上传角色视频
        
        支持两种方式：
        1. 通过字符视频URL上传
        2. 直接上传视频文件（通过character_video参数）
        """
        self.initialize(api_key)
        
        # 处理时间戳
        try:
            # 验证时间戳格式
            ts_parts = timestamps.split(",")
            if len(ts_parts) != 2:
                raise ValueError("时间戳格式应为 '开始时间,结束时间'")
            
            start_ts = float(ts_parts[0].strip())
            end_ts = float(ts_parts[1].strip())
            
            if start_ts < 0 or end_ts < 0:
                raise ValueError("时间戳不能为负数")
            if start_ts >= end_ts:
                raise ValueError("开始时间必须小于结束时间")
                
            # 格式化为字符串，保留两位小数
            formatted_timestamps = f"{start_ts:.2f},{end_ts:.2f}"
            
        except ValueError as e:
            return (f"时间戳格式错误: {str(e)}", "")
        
        # 确定视频URL
        final_video_url = character_video_url.strip()
        
        # 如果提供了视频文件且URL为空，尝试上传视频文件
        if character_video is not None and not final_video_url:
            final_video_url = self._upload_video_file(character_video, api_key)
            if not final_video_url:
                return ("视频文件上传失败", "")
        
        # 验证URL
        if not final_video_url:
            return ("必须提供角色视频URL或视频文件", "")
        
        # 准备请求体
        payload = {
            "url": final_video_url,
            "timestamps": formatted_timestamps,
            "shutProgress": shut_progress
        }
        
        # 发送请求
        result, error = self.client.make_request(
            endpoint="/v1/video/sora-upload-character",
            payload=payload,
            response_mode=response_mode,
            webhook_url=webhook_url,
            use_cn_endpoint=use_cn_endpoint,
            timeout=300
        )
        
        if error:
            print(f"上传角色视频错误: {error}")
            return self.get_empty_response(self.RETURN_TYPES)
        
        # 处理结果
        character_id = ""
        task_id = ""
        
        if response_mode == "polling":
            # 轮询模式，只返回任务ID
            if result and result.get('code') == 0:
                task_id = result['data'].get('id', '')
                print(f"角色上传任务创建成功。任务ID: {task_id}")
        else:
            # 流式或webhook模式
            if result and result.get('status') == 'succeeded':
                results = result.get('results', [])
                if results:
                    character_id = results[0].get('character_id', '')
                    task_id = result.get('id', '')
                    
                    if character_id:
                        print(f"角色上传成功！角色ID: {character_id}")
                    if task_id:
                        print(f"任务ID: {task_id}")
        
        # 如果没有获取到结果，尝试从不同字段提取
        if not character_id or not task_id:
            character_id, task_id = self._extract_ids_from_response(result)
        
        return (character_id or "", task_id or "")
    
    def _upload_video_file(self, video, api_key: str) -> str:
        """
        上传视频文件到CDN并返回URL
        
        Args:
            video: VideoAdapter对象或视频文件路径
            api_key: API密钥
            
        Returns:
            视频URL或空字符串
        """
        try:
            from .utils import upload_video_to_apimart_cdn
            
            # 如果video是VideoAdapter对象
            if hasattr(video, 'path') and video.path:
                video_path = video.path
            elif isinstance(video, str):
                video_path = video
            else:
                print("不支持的视频格式")
                return ""
            
            # 上传视频到CDN
            cdn_url = upload_video_to_apimart_cdn(video_path, api_key)
            if cdn_url:
                print(f"视频上传到CDN成功: {cdn_url}")
                return cdn_url
            
            return ""
            
        except Exception as e:
            print(f"视频文件上传失败: {str(e)}")
            return ""
    
    def _extract_ids_from_response(self, response: Dict[str, Any]) -> Tuple[str, str]:
        """
        从API响应中提取角色ID和任务ID
        
        Args:
            response: API响应数据
            
        Returns:
            Tuple[角色ID, 任务ID]
        """
        if not response or not isinstance(response, dict):
            return "", ""
        
        character_id = ""
        task_id = ""
        
        # 提取角色ID
        character_id_fields = [
            "character_id",
            "characterId",
            "character",
            "id",  # 有时角色ID就是id字段
        ]
        
        # 提取任务ID
        task_id_fields = [
            "task_id",
            "taskId",
            "id",
            "request_id",
            "requestId",
        ]
        
        # 深度搜索函数
        def deep_search(obj, target_fields, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return ""
            
            if isinstance(obj, dict):
                # 先检查当前层级
                for field in target_fields:
                    if field in obj and isinstance(obj[field], str) and obj[field].strip():
                        return obj[field].strip()
                
                # 递归搜索
                for value in obj.values():
                    result = deep_search(value, target_fields, max_depth, current_depth + 1)
                    if result:
                        return result
            
            elif isinstance(obj, list):
                for item in obj:
                    result = deep_search(item, target_fields, max_depth, current_depth + 1)
                    if result:
                        return result
            
            return ""
        
        # 从响应中提取
        character_id = deep_search(response, character_id_fields)
        task_id = deep_search(response, task_id_fields)
        
        # 特殊处理：有时角色ID在results数组中
        if not character_id and "results" in response and isinstance(response["results"], list):
            for result_item in response["results"]:
                if isinstance(result_item, dict):
                    for field in character_id_fields:
                        if field in result_item and result_item[field]:
                            character_id = str(result_item[field])
                            break
                if character_id:
                    break
        
        # 特殊处理：有时任务ID在data字段中
        if not task_id and "data" in response and isinstance(response["data"], dict):
            for field in task_id_fields:
                if field in response["data"] and response["data"][field]:
                    task_id = str(response["data"][field])
                    break
        
        return character_id or "", task_id or ""
    
    def validate_timestamps(self, timestamps_str: str, video_duration: float = None) -> Tuple[bool, str, float, float]:
        """
        验证时间戳格式和有效性
        
        Args:
            timestamps_str: 时间戳字符串，格式如 "0,3"
            video_duration: 视频时长（秒），可选
            
        Returns:
            Tuple[是否有效, 错误信息, 开始时间, 结束时间]
        """
        try:
            parts = timestamps_str.split(",")
            if len(parts) != 2:
                return False, "时间戳格式应为 '开始时间,结束时间'", 0, 0
            
            start_time = float(parts[0].strip())
            end_time = float(parts[1].strip())
            
            # 基本验证
            if start_time < 0:
                return False, "开始时间不能为负数", start_time, end_time
            if end_time < 0:
                return False, "结束时间不能为负数", start_time, end_time
            if start_time >= end_time:
                return False, "开始时间必须小于结束时间", start_time, end_time
            
            # 如果提供了视频时长，验证是否超出范围
            if video_duration is not None:
                if start_time > video_duration:
                    return False, f"开始时间({start_time}s)超出视频时长({video_duration}s)", start_time, end_time
                if end_time > video_duration:
                    return False, f"结束时间({end_time}s)超出视频时长({video_duration}s)", start_time, end_time
                
                # 验证片段长度
                segment_length = end_time - start_time
                if segment_length < 1.0:
                    return False, f"视频片段太短({segment_length:.1f}s)，至少需要1秒", start_time, end_time
                if segment_length > 10.0:
                    return False, f"视频片段太长({segment_length:.1f}s)，最多10秒", start_time, end_time
            
            return True, "", start_time, end_time
            
        except ValueError:
            return False, "时间戳必须为数字", 0, 0
        except Exception as e:
            return False, f"验证时间戳时出错: {str(e)}", 0, 0


class GrsaiSoraCreateCharacterNodeZV(GrsaiBaseNode):
    """从原视频创建角色节点（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "pid": ("STRING", {"default": "", "multiline": False}),
                "timestamps": ("STRING", {"default": "0,3", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "task_id")
    FUNCTION = "create_character"
    CATEGORY = "Sora Video"
    
    def create_character(self, api_key, pid, timestamps, use_cn_endpoint, 
                        response_mode, webhook_url, shut_progress):
        """
        从现有视频的PID创建角色
        """
        self.initialize(api_key)
        
        # 验证PID
        pid = pid.strip()
        if not pid:
            return ("PID不能为空", "")
        
        # 处理时间戳
        try:
            ts_parts = timestamps.split(",")
            if len(ts_parts) != 2:
                raise ValueError("时间戳格式应为 '开始时间,结束时间'")
            
            start_ts = float(ts_parts[0].strip())
            end_ts = float(ts_parts[1].strip())
            
            if start_ts < 0 or end_ts < 0:
                raise ValueError("时间戳不能为负数")
            if start_ts >= end_ts:
                raise ValueError("开始时间必须小于结束时间")
                
            formatted_timestamps = f"{start_ts:.2f},{end_ts:.2f}"
            
        except ValueError as e:
            return (f"时间戳格式错误: {str(e)}", "")
        
        # 准备请求体
        payload = {
            "pid": pid,
            "timestamps": formatted_timestamps,
            "shutProgress": shut_progress
        }
        
        # 发送请求
        result, error = self.client.make_request(
            endpoint="/v1/video/sora-create-character",
            payload=payload,
            response_mode=response_mode,
            webhook_url=webhook_url,
            use_cn_endpoint=use_cn_endpoint,
            timeout=300
        )
        
        if error:
            print(f"从视频创建角色错误: {error}")
            return self.get_empty_response(self.RETURN_TYPES)
        
        # 处理结果
        character_id = ""
        task_id = ""
        
        if response_mode == "polling":
            # 轮询模式，只返回任务ID
            if result and result.get('code') == 0:
                task_id = result['data'].get('id', '')
                print(f"角色创建任务创建成功。任务ID: {task_id}")
        else:
            # 流式或webhook模式
            if result and result.get('status') == 'succeeded':
                results = result.get('results', [])
                if results:
                    character_id = results[0].get('character_id', '')
                    task_id = result.get('id', '')
                    
                    if character_id:
                        print(f"角色创建成功！角色ID: {character_id}")
                    if task_id:
                        print(f"任务ID: {task_id}")
        
        # 如果没有获取到结果，尝试从不同字段提取
        if not character_id or not task_id:
            character_id, task_id = self._extract_ids_from_response(result)
        
        return (character_id or "", task_id or "")
    
    def _extract_ids_from_response(self, response: Dict[str, Any]) -> Tuple[str, str]:
        """
        从API响应中提取角色ID和任务ID
        """
        if not response or not isinstance(response, dict):
            return "", ""
        
        character_id = ""
        task_id = ""
        
        # 提取角色ID
        if "results" in response and isinstance(response["results"], list) and response["results"]:
            result_item = response["results"][0]
            if isinstance(result_item, dict):
                character_id = result_item.get("character_id", "")
        
        # 提取任务ID
        task_id = response.get("id", "")
        if not task_id and "data" in response and isinstance(response["data"], dict):
            task_id = response["data"].get("id", "")
        
        return character_id or "", task_id or ""


class GrsaiNanoBananaGeminiNodeZV(GrsaiBaseNode):
    """Nano Banana Gemini格式接口节点（重构版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["nano-banana-fast"], {"default": "nano-banana-fast"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "reference_image_urls": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "task_id", "content")
    FUNCTION = "generate_image_gemini"
    CATEGORY = "Nano Banana"
    
    def generate_image_gemini(self, api_key, prompt, model, use_cn_endpoint,
                             reference_images=None, reference_image_urls=""):
        """
        使用Gemini格式接口生成图像
        """
        self.initialize(api_key)
        
        base_url = self.client.get_base_url(use_cn_endpoint)
        endpoint = f"/v1beta/models/{model}:streamGenerateContent"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备Gemini格式的请求体
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
        
        # 处理参考图像（Base64格式，不带前缀）
        if reference_images is not None:
            base64_images = self.client.image_to_base64(reference_images, include_prefix=False)
            if base64_images:
                if isinstance(base64_images, list):
                    for img_base64 in base64_images:
                        contents[0]["parts"].append({
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        })
                else:
                    contents[0]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_images
                        }
                    })
        
        # 处理图像URL
        if reference_image_urls:
            url_list = [url.strip() for url in reference_image_urls.split(",") if url.strip()]
            for url in url_list:
                try:
                    image_base64 = self._download_and_encode_image(url)
                    if image_base64:
                        # 根据URL扩展名确定MIME类型
                        mime_type = "image/jpeg" if url.lower().endswith('.jpg') or url.lower().endswith('.jpeg') else "image/png"
                        contents[0]["parts"].append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        })
                except Exception as e:
                    print(f"处理图像URL失败 {url}: {str(e)}")
        
        payload = {
            "contents": contents,
            "generation_config": {
                "temperature": 0.4,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
                "stopSequences": []
            },
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        print(f"发送Gemini格式请求到Nano Banana API...")
        
        try:
            # 发送请求（使用流式，因为Gemini格式通常返回流式响应）
            url = f"{base_url}{endpoint}"
            response = self.client.session.post(url, json=payload, headers=headers, stream=True, timeout=300)
            
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码 {response.status_code}: {response.text}"
                print(f"错误: {error_msg}")
                return self.get_empty_response(self.RETURN_TYPES)
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 去掉 'data: ' 前缀
                            if data_str.strip():
                                data = json.loads(data_str)
                                
                                # 解析Gemini格式响应
                                if 'candidates' in data and data['candidates']:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                full_response += part['text']
                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"解析JSON流失败: {e}")
                        continue
            
            print(f"Nano Banana Gemini API响应: {full_response[:200]}...")
            
            # 注意：Gemini格式的响应可能不包含图像URL，这里返回响应文本作为content
            # 尝试从响应中提取图像URL（如果有）
            image_url = ""
            task_id = ""
            
            # 简单的URL提取逻辑
            import re
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            urls = re.findall(url_pattern, full_response)
            if urls:
                image_url = urls[0]  # 取第一个URL
            
            return (image_url, task_id, full_response)
                
        except Exception as e:
            print(f"Nano Banana Gemini图像生成错误: {str(e)}")
            return self.get_empty_response(self.RETURN_TYPES)
    
    def _download_and_encode_image(self, url: str) -> Optional[str]:
        """下载图像并编码为Base64"""
        try:
            import base64
            import requests
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 编码为Base64
            img_base64 = base64.b64encode(response.content).decode()
            return img_base64
            
        except Exception as e:
            print(f"从 {url} 下载图像失败: {str(e)}")
            return None


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GrsaiSoraVideoNodeZV": GrsaiSoraVideoNodeZV,
    "GrsaiSoraUploadCharacterNodeZV": GrsaiSoraUploadCharacterNodeZV,  # 需要按相同模式重构
    "GrsaiSoraCreateCharacterNodeZV": GrsaiSoraCreateCharacterNodeZV,  # 需要按相同模式重构
    "GrsaiResultNodeZV": GrsaiResultNodeZV,
    "GrsaiNanoBananaNodeZV": GrsaiNanoBananaNodeZV,
    "GrsaiNanoBananaGeminiNodeZV": GrsaiNanoBananaGeminiNodeZV,  # 需要按相同模式重构
    "GrsaiLoadImageFromPathNodeZV": GrsaiLoadImageFromPathNodeZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiSoraVideoNodeZV": "Grsai Sora Video Generator",
    "GrsaiSoraUploadCharacterNodeZV": "Grsai Sora Upload Character",
    "GrsaiSoraCreateCharacterNodeZV": "Grsai Sora Create Character from Video",
    "GrsaiResultNodeZV": "Grsai Get Result",
    "GrsaiNanoBananaNodeZV": "Grsai Nano Banana Image Generator",
    "GrsaiNanoBananaGeminiNodeZV": "Grsai Nano Banana Gemini Format",
    "GrsaiLoadImageFromPathNodeZV": "Grsai Load Image from Path",
}