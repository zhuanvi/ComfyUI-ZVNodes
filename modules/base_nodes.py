# base_nodes.py
import time
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import json
import requests
import torch
from PIL import Image
import io
import base64
import numpy as np

from .config import APIMartConfig
from .api_client import APIMartClient
from .task_manager import TaskManager
from .response_parser import ResponseParser
from .utils import batch_upload_image_to_apimart_cdn, image_to_temp_png

class BaseAPIMartNode(ABC):
    """API Mart 节点基类"""
    
    CATEGORY = "ZVNodes/apimart"
    
    def __init__(self):
        self.client = None
        self.task_manager = None
        self.parser = ResponseParser()
    
    def initialize(self, api_key: str = ""):
        """初始化客户端和任务管理器"""
        if not hasattr(self, '_initialized') or not self._initialized:
            self.client = APIMartClient(api_key)
            
            # 初始化任务管理器
            import os
            NODE_DIR = os.path.dirname(__file__)
            task_file = os.path.join(NODE_DIR, "apimart_task_history.json")
            self.task_manager = TaskManager(task_file)
            
            self._initialized = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    @abstractmethod
    def process(self, **kwargs) -> Tuple:
        """处理节点逻辑"""
        pass
    
    def handle_error(self, error: Exception, context: str = "") -> Tuple:
        """统一错误处理"""
        error_msg = f"Error in {self.__class__.__name__}: {str(error)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        # 根据节点类型返回适当的空值
        if hasattr(self, 'RETURN_TYPES'):
            return_types = self.RETURN_TYPES
            return tuple([self._get_empty_value(rt) for rt in return_types])
        
        return ()
    
    def _get_empty_value(self, return_type: str) -> Any:
        """根据返回类型获取空值"""
        if return_type == "STRING":
            return ""
        elif return_type == "IMAGE":
            try:
                import torch
                return torch.zeros((0, 3, 512, 512))
            except Exception:
                return None
        elif return_type == "VIDEO":
            from .utils import VideoAdapter
            return VideoAdapter(None)
        else:
            return None
    
    def _handle_image_upload_and_submit(
        self,
        image,
        prompt: str,
        api_key: str,
        model: str,
        aspect_ratio: str,
        duration: int,
        max_side: int = 1024,
        task_type: str = "image2video"
    ) -> Tuple[Dict[str, Any], str, Optional[str], str]:
        """
        处理图像上传并提交的通用方法（修复版）
        
        Args:
            image: 输入图像
            prompt: 提示词
            api_key: API密钥
            model: 模型名称
            aspect_ratio: 宽高比
            duration: 视频时长
            max_side: 图像最大边长
            task_type: 任务类型
            
        Returns:
            Tuple[response_body, method_description, task_id, error_message]
        """
        try:
            # 1. 尝试CDN上传
            cdn_urls = batch_upload_image_to_apimart_cdn(image, api_key, max_side=max_side)
            
            if cdn_urls and isinstance(cdn_urls, list) and len(cdn_urls) > 0:
                # 使用CDN URLs提交
                return self._submit_with_cdn_urls(
                    cdn_urls, prompt, model, aspect_ratio, duration, task_type, api_key
                )
            
            # 2. 兜底：multipart上传
            return self._submit_with_multipart(
                image, prompt, api_key, model, aspect_ratio, duration, max_side, task_type
            )
            
        except Exception as e:
            error_msg = f"图像上传和提交失败: {str(e)}"
            print(f"Error in _handle_image_upload_and_submit: {error_msg}")
            return {"error": error_msg}, "failed", None, error_msg
    
    def _submit_with_cdn_urls(
        self,
        cdn_urls: List[str],
        prompt: str,
        model: str,
        aspect_ratio: str,
        duration: int,
        task_type: str = "image2video",
        api_key: str = ""
    ) -> Tuple[Dict[str, Any], str, Optional[str], str]:
        """使用CDN URLs提交（修复版）"""
        try:
            # 先尝试 image_urls 字段
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "image_urls": cdn_urls,
            }
            
            # 确保客户端已初始化
            if not self.client and api_key:
                self.initialize(api_key)
            elif not self.client:
                return {"error": "客户端未初始化"}, "CDN外链", None, "客户端未初始化"
            
            code, body = self.client.submit_video_generation(payload)
            task_id = self.parser.extract_task_id(body) if body else None
            
            # 如果失败，尝试使用单一 url 字段
            if code != 200 or not task_id:
                payload_retry = {
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "url": cdn_urls[0] if cdn_urls else "",
                }
                
                code, body = self.client.submit_video_generation(payload_retry)
                task_id = self.parser.extract_task_id(body) if body else task_id  # 保持原有的task_id
            
            # 记录日志以便调试
            print(f"CDN提交 - 状态码: {code}, Task ID: {task_id}, 响应: {body}")
            
            # 如果成功且获取到task_id，保存到任务管理器
            if task_id and code == 200:
                try:
                    self.task_manager.add_task(task_id, task_type)
                    print(f"任务已保存到管理器: {task_id}")
                except Exception as e:
                    print(f"保存任务到管理器失败: {str(e)}")
            
            return body, "CDN外链", task_id, ""
            
        except Exception as e:
            error_msg = f"CDN提交失败: {str(e)}"
            print(f"Error in _submit_with_cdn_urls: {error_msg}")
            return {"error": error_msg}, "CDN外链", None, error_msg
    
    def _submit_with_multipart(
        self,
        image,
        prompt: str,
        api_key: str,
        model: str,
        aspect_ratio: str,
        duration: int,
        max_side: int = 1024,
        task_type: str = "image2video"
    ) -> Tuple[Dict[str, Any], str, Optional[str], str]:
        """使用multipart上传（修复版）"""
        temp_png = None
        try:
            temp_png = image_to_temp_png(image, max_side=max_side)
            if not temp_png:
                return {"error": "图片上传失败且本地PNG生成失败"}, "failed", None, "图片上传失败且本地PNG生成失败"
            
            form_data = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }
            
            file_field = os.getenv("APIMART_FILE_FIELD", "image").strip() or "image"
            
            # 确保客户端已初始化
            if not self.client and api_key:
                self.initialize(api_key)
            elif not self.client:
                return {"error": "客户端未初始化"}, "multipart", None, "客户端未初始化"
            
            # 打开文件并上传
            with open(temp_png, "rb") as file_obj:
                files = {
                    file_field: (
                        os.path.basename(temp_png) or "image.png",
                        file_obj,
                        "image/png"
                    )
                }
                
                code, body = self.client.submit_video_generation(
                    form_data,
                    use_multipart=True,
                    files=files
                )
                
                task_id = self.parser.extract_task_id(body) if body else None
                
                # 记录日志以便调试
                print(f"Multipart提交 - 状态码: {code}, Task ID: {task_id}, 响应: {body}")
                
                # 如果成功且获取到task_id，保存到任务管理器
                if task_id and code == 200:
                    try:
                        self.task_manager.add_task(task_id, task_type)
                        print(f"任务已保存到管理器: {task_id}")
                    except Exception as e:
                        print(f"保存任务到管理器失败: {str(e)}")
                
                return body, f"multipart上传({file_field})", task_id, ""
                
        except Exception as e:
            error_msg = f"multipart上传失败: {str(e)}"
            print(f"Error in _submit_with_multipart: {error_msg}")
            return {"error": error_msg}, "multipart", None, error_msg
            
        finally:
            # 清理临时文件
            if temp_png and os.path.exists(temp_png):
                try:
                    os.unlink(temp_png)
                except Exception:
                    pass
    
    def _poll_task_status(
        self,
        task_id: str,
        api_key: str = "",
        max_retries: int = 12,
        retry_interval: int = 5,
        extract_func: callable = None,
        poll_until_success: bool = True,
        timeout: int = 30
    ) -> Tuple[Optional[Any], Optional[str], int, Optional[Dict[str, Any]]]:
        """
        轮询任务状态
        
        Args:
            task_id: 任务ID
            api_key: API密钥（如果未初始化客户端）
            max_retries: 最大重试次数
            retry_interval: 重试间隔（秒）
            extract_func: 提取结果的函数，默认使用extract_video_url
            poll_until_success: 是否轮询直到成功
            timeout: 每次请求的超时时间
            
        Returns:
            Tuple[提取的结果, 状态, 重试次数, 完整响应数据]
        """
        try:
            # 确保客户端已初始化
            if not self.client and api_key:
                self.initialize(api_key)
            elif not self.client:
                return None, "client_not_initialized", 0, None
            
            if extract_func is None:
                extract_func = self.parser.extract_video_url
            
            retry_count = 0
            last_response = None
            
            while retry_count < max_retries:
                try:
                    # 查询任务状态
                    code, body = self.client.query_task(task_id, poll_until_done=False)
                    last_response = body
                    
                    if code == 200:
                        # 提取状态信息
                        status = self.parser.extract_status(body)
                        
                        # 尝试提取结果
                        result = extract_func(body)
                        
                        # 如果成功提取到结果，立即返回
                        if result:
                            return result, status or "completed", retry_count, body
                        
                        # 检查任务是否已完成但无结果
                        if status in ["completed", "succeeded"]:
                            # 任务已完成但没有提取到结果
                            return None, status, retry_count, body
                        
                        # 任务失败
                        if status in ["failed", "cancelled", "error"]:
                            return None, status, retry_count, body
                        
                        # 任务仍在处理中
                        if poll_until_success:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"任务处理中，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                                time.sleep(retry_interval)
                                continue
                            else:
                                return None, "timeout", retry_count, body
                        else:
                            # 不轮询，直接返回当前状态
                            return None, status or "processing", retry_count, body
                    
                    elif code == 401:
                        # 认证错误
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"HTTP 401 - 等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                            time.sleep(retry_interval)
                            continue
                        else:
                            return None, "auth_error", retry_count, body
                    
                    elif code == 404:
                        # 任务不存在
                        return None, "not_found", retry_count, body
                    
                    else:
                        # 其他错误
                        retry_count += 1
                        if retry_count < max_retries and poll_until_success:
                            print(f"HTTP {code} - 等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                            time.sleep(retry_interval)
                            continue
                        else:
                            return None, f"http_error_{code}", retry_count, body
                            
                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"请求超时，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        return None, "timeout_error", retry_count, last_response
                        
                except requests.exceptions.ConnectionError:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"网络连接错误，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        return None, "connection_error", retry_count, last_response
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"轮询异常: {str(e)}，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        return None, f"exception: {str(e)}", retry_count, last_response
            
            # 达到最大重试次数
            return None, "max_retries_exceeded", max_retries, last_response
            
        except Exception as e:
            print(f"轮询任务状态异常: {str(e)}")
            return None, f"polling_error: {str(e)}", 0, None
    
    def _poll_task_status_with_progress(
        self,
        task_id: str,
        api_key: str = "",
        max_retries: int = 12,
        retry_interval: int = 5,
        extract_func: callable = None,
        progress_callback: callable = None,
        timeout: int = 30
    ) -> Tuple[Optional[Any], Optional[str], int, Optional[int], Optional[Dict[str, Any]]]:
        """
        带进度报告的轮询任务状态
        
        Args:
            task_id: 任务ID
            api_key: API密钥
            max_retries: 最大重试次数
            retry_interval: 重试间隔
            extract_func: 提取结果的函数
            progress_callback: 进度回调函数
            timeout: 超时时间
            
        Returns:
            Tuple[提取的结果, 状态, 重试次数, 进度百分比, 完整响应数据]
        """
        try:
            if not self.client and api_key:
                self.initialize(api_key)
            elif not self.client:
                return None, "client_not_initialized", 0, 0, None
            
            if extract_func is None:
                extract_func = self.parser.extract_video_url
            
            retry_count = 0
            last_response = None
            last_progress = 0
            
            while retry_count < max_retries:
                try:
                    code, body = self.client.query_task(task_id, poll_until_done=False)
                    last_response = body
                    
                    if code == 200:
                        # 提取状态和进度
                        status = self.parser.extract_status(body)
                        progress = self._extract_progress(body)
                        
                        # 更新进度
                        if progress is not None and progress != last_progress:
                            last_progress = progress
                            if progress_callback:
                                progress_callback(progress, status)
                            else:
                                print(f"任务进度: {progress}%, 状态: {status}")
                        
                        # 尝试提取结果
                        result = extract_func(body)
                        
                        # 检查任务状态
                        if result:
                            return result, status or "completed", retry_count, progress, body
                        
                        if status in ["completed", "succeeded"]:
                            return None, status, retry_count, progress, body
                        
                        if status in ["failed", "cancelled", "error"]:
                            return None, status, retry_count, progress, body
                        
                        # 继续轮询
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"任务处理中 ({progress}%)，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                            time.sleep(retry_interval)
                            continue
                        else:
                            return None, "timeout", retry_count, progress, body
                    
                    else:
                        # 处理错误
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"HTTP {code}，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                            time.sleep(retry_interval)
                            continue
                        else:
                            return None, f"http_error_{code}", retry_count, last_progress, body
                            
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"轮询异常: {str(e)}，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        return None, f"exception: {str(e)}", retry_count, last_progress, last_response
            
            return None, "max_retries_exceeded", max_retries, last_progress, last_response
            
        except Exception as e:
            print(f"带进度轮询异常: {str(e)}")
            return None, f"progress_polling_error: {str(e)}", 0, 0, None
    
    def _extract_progress(self, response: Dict[str, Any]) -> Optional[int]:
        """从响应中提取进度信息"""
        if not isinstance(response, dict):
            return None
        
        # 尝试不同的进度字段
        progress_fields = ["progress", "percentage", "complete_percent", "completion"]
        
        for field in progress_fields:
            if field in response and isinstance(response[field], (int, float)):
                value = int(response[field])
                if 0 <= value <= 100:
                    return value
        
        # 检查嵌套字段
        for nested in ["data", "output", "result"]:
            if isinstance(response.get(nested), dict):
                for field in progress_fields:
                    value = response[nested].get(field)
                    if isinstance(value, (int, float)):
                        value = int(value)
                        if 0 <= value <= 100:
                            return value
        
        return None
    
    def _extract_error_details(self, response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """从响应中提取错误详细信息"""
        if not isinstance(response, dict):
            return None, None
        
        error_message = None
        error_code = None
        
        # 提取错误消息
        for field in ["error", "message", "detail", "reason", "error_message"]:
            value = response.get(field)
            if value:
                if isinstance(value, dict):
                    error_message = value.get("message", str(value))
                else:
                    error_message = str(value)
                break
        
        # 提取错误代码
        for field in ["code", "error_code", "status_code"]:
            value = response.get(field)
            if value is not None:
                error_code = str(value)
                break
        
        return error_message, error_code
    
    def _wait_for_task_completion(
        self,
        task_id: str,
        api_key: str = "",
        max_wait_time: int = 300,
        poll_interval: int = 5,
        extract_func: callable = None,
        silent: bool = False
    ) -> Tuple[Optional[Any], Optional[str], bool]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            api_key: API密钥
            max_wait_time: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            extract_func: 提取结果的函数
            silent: 是否静默模式（不打印日志）
            
        Returns:
            Tuple[提取的结果, 最终状态, 是否成功]
        """
        start_time = time.time()
        poll_count = 0
        
        if not silent:
            print(f"开始等待任务完成: {task_id}")
        
        while time.time() - start_time < max_wait_time:
            poll_count += 1
            
            # 轮询任务状态
            result, status, retry_count, full_response = self._poll_task_status(
                task_id=task_id,
                api_key=api_key,
                max_retries=1,  # 只尝试一次，由外部循环控制
                retry_interval=poll_interval,
                extract_func=extract_func,
                poll_until_success=False
            )
            
            if not silent:
                print(f"轮询 {poll_count}: 状态={status}")
            
            # 检查是否完成
            if status in ["completed", "succeeded"]:
                if result:
                    if not silent:
                        print(f"任务完成，成功提取结果")
                    return result, status, True
                else:
                    if not silent:
                        print(f"任务完成，但未提取到结果")
                    return None, status, False
            
            # 检查是否失败
            if status in ["failed", "cancelled", "error", "auth_error", "not_found"]:
                if not silent:
                    print(f"任务失败，状态: {status}")
                return None, status, False
            
            # 仍在处理中，等待下一轮
            time.sleep(poll_interval)
        
        # 超时
        if not silent:
            print(f"等待任务超时: {max_wait_time}秒")
        return None, "timeout", False
    
    def _download_media_with_progress(
        self,
        url: str,
        output_path: str,
        task_id: str = "",
        chunk_size: int = 8192,
        progress_callback: callable = None
    ) -> Tuple[bool, str, Optional[int]]:
        """
        下载媒体文件并显示进度
        
        Args:
            url: 下载URL
            output_path: 输出路径
            task_id: 任务ID（用于日志）
            chunk_size: 块大小
            progress_callback: 进度回调函数
            
        Returns:
            Tuple[是否成功, 消息, 文件大小(字节)]
        """
        try:
            import requests
            import os
            
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
            
            # 获取文件大小
            response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
            total_size = 0
            
            if 'Content-Length' in response.headers:
                total_size = int(response.headers['Content-Length'])
            
            # 开始下载
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            downloaded_size = 0
            last_percent = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 计算并报告进度
                        if total_size > 0:
                            percent = int((downloaded_size / total_size) * 100)
                            if percent != last_percent and (percent % 10 == 0 or percent == 100):
                                last_percent = percent
                                msg = f"下载进度: {percent}% ({downloaded_size}/{total_size} bytes)"
                                if task_id:
                                    msg = f"[{task_id}] {msg}"
                                
                                if progress_callback:
                                    progress_callback(percent, msg)
                                else:
                                    print(msg)
            
            # 验证文件大小
            actual_size = os.path.getsize(output_path)
            if total_size > 0 and actual_size != total_size:
                return False, f"文件大小不匹配: 期望{total_size}字节，实际{actual_size}字节", actual_size
            
            return True, f"下载完成: {output_path} ({actual_size}字节)", actual_size
            
        except Exception as e:
            error_msg = f"下载失败: {str(e)}"
            if task_id:
                error_msg = f"[{task_id}] {error_msg}"
            return False, error_msg, None
    
    def _batch_poll_tasks(
        self,
        task_ids: List[str],
        api_key: str = "",
        max_retries_per_task: int = 3,
        poll_interval: int = 2,
        extract_func: callable = None,
        parallel: bool = False
    ) -> Dict[str, Tuple[Optional[Any], Optional[str]]]:
        """
        批量轮询多个任务
        
        Args:
            task_ids: 任务ID列表
            api_key: API密钥
            max_retries_per_task: 每个任务的最大重试次数
            poll_interval: 轮询间隔
            extract_func: 提取结果的函数
            parallel: 是否并行轮询（需要线程支持）
            
        Returns:
            字典: {task_id: (结果, 状态)}
        """
        results = {}
        
        if parallel:
            # 并行轮询（简单实现）
            import threading
            
            def poll_single(task_id):
                result, status, _, _ = self._poll_task_status(
                    task_id=task_id,
                    api_key=api_key,
                    max_retries=max_retries_per_task,
                    retry_interval=poll_interval,
                    extract_func=extract_func
                )
                results[task_id] = (result, status)
            
            threads = []
            for task_id in task_ids:
                thread = threading.Thread(target=poll_single, args=(task_id,))
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()
        else:
            # 串行轮询
            for task_id in task_ids:
                result, status, _, _ = self._poll_task_status(
                    task_id=task_id,
                    api_key=api_key,
                    max_retries=max_retries_per_task,
                    retry_interval=poll_interval,
                    extract_func=extract_func
                )
                results[task_id] = (result, status)
        
        return results
    
    def _monitor_tasks_with_timeout(
        self,
        task_ids: List[str],
        api_key: str = "",
        timeout: int = 600,  # 10分钟
        poll_interval: int = 10,
        completion_callback: callable = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        监控多个任务直到超时
        
        Args:
            task_ids: 任务ID列表
            api_key: API密钥
            timeout: 总超时时间（秒）
            poll_interval: 轮询间隔
            completion_callback: 完成回调函数
            
        Returns:
            字典: {task_id: 任务信息}
        """
        import time
        
        start_time = time.time()
        task_status = {task_id: {"status": "pending", "result": None} for task_id in task_ids}
        completed_tasks = set()
        
        while time.time() - start_time < timeout:
            # 检查未完成的任务
            pending_tasks = [tid for tid in task_ids if tid not in completed_tasks]
            
            if not pending_tasks:
                break  # 所有任务都完成了
            
            # 轮询未完成的任务
            for task_id in pending_tasks:
                result, status, _, full_response = self._poll_task_status(
                    task_id=task_id,
                    api_key=api_key,
                    max_retries=1,
                    retry_interval=0,
                    poll_until_success=False
                )
                
                task_status[task_id]["status"] = status
                task_status[task_id]["result"] = result
                task_status[task_id]["last_check"] = time.time() - start_time
                
                # 检查任务是否完成
                if status in ["completed", "succeeded", "failed", "cancelled"]:
                    completed_tasks.add(task_id)
                    
                    if completion_callback:
                        completion_callback(task_id, status, result)
            
            # 如果还有未完成的任务，等待下一轮
            if len(pending_tasks) > len(completed_tasks):
                time.sleep(poll_interval)
        
        return task_status
    
    def _download_with_retries(
        self,
        url: str,
        target_path: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """带重试的下载函数"""
        import requests
        import os
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
        backoff = 5.0
        
        for attempt in range(1, max_retries + 1):
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    if 400 <= r.status_code < 500:
                        return False, f"客户端错误 {r.status_code}，停止重试"
                    r.raise_for_status()
                    with open(target_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return True, f"下载成功: {target_path}"
            except Exception as e:
                if attempt == max_retries:
                    return False, f"下载失败: {e}"
                time.sleep(backoff)
                backoff *= 2
        
        return False, "下载失败: 达到最大重试次数"

class GrsaiBaseClient(ABC):
    """Grsai API基础客户端"""
    
    def __init__(self, api_key: str = ""):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
        })
    
    def get_base_url(self, use_cn_endpoint: bool) -> str:
        """获取基础URL"""
        return self.base_url_cn if use_cn_endpoint else self.base_url_overseas
    
    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    @staticmethod
    def image_to_base64(image_tensor: torch.Tensor, include_prefix: bool = True) -> Union[str, List[str]]:
        """将图像张量转换为Base64"""
        if image_tensor is None:
            return None
            
        def process_single_tensor(tensor):
            # 转换为PIL图像
            image_np = tensor.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            image = Image.fromarray(image_np)
            
            # 转换为Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            if include_prefix:
                return f"data:image/png;base64,{img_str}"
            return img_str
        
        # 处理多个图像的情况
        if len(image_tensor.shape) == 4:
            return [process_single_tensor(image_tensor[i]) for i in range(image_tensor.shape[0])]
        else:
            return process_single_tensor(image_tensor)
    
    def make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        response_mode: str = "stream",
        webhook_url: str = "",
        use_cn_endpoint: bool = True,
        timeout: int = 30
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            payload: 请求数据
            response_mode: 响应模式 (stream, webhook, polling)
            webhook_url: Webhook URL
            use_cn_endpoint: 是否使用国内端点
            timeout: 超时时间
            
        Returns:
            Tuple[响应数据, 错误信息]
        """
        base_url = self.get_base_url(use_cn_endpoint)
        url = f"{base_url}{endpoint}"
        
        # 处理响应模式
        if response_mode == "webhook" and webhook_url:
            payload["webHook"] = webhook_url
        elif response_mode == "polling":
            payload["webHook"] = "-1"
        
        headers = self.get_headers()
        
        try:
            if response_mode == "stream":
                return self._handle_stream_request(url, payload, headers, timeout)
            else:
                return self._handle_normal_request(url, payload, headers, timeout)
                
        except requests.exceptions.Timeout:
            return None, "请求超时"
        except requests.exceptions.ConnectionError:
            return None, "网络连接错误"
        except requests.exceptions.RequestException as e:
            return None, f"请求错误: {str(e)}"
        except Exception as e:
            return None, f"未知错误: {str(e)}"
    
    def _handle_stream_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """处理流式请求"""
        try:
            response = self.session.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            last_data = None
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        progress = data.get('progress', 0)
                        status = data.get('status', 'unknown')
                        print(f"API progress: {progress}% - Status: {status}")
                        
                        if data.get('status') == 'succeeded':
                            last_data = data
                    except json.JSONDecodeError:
                        continue
            
            return last_data, None
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误 {e.response.status_code}: {e.response.text}"
            print(f"Error: {error_msg}")
            return None, error_msg
    
    def _handle_normal_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """处理普通请求"""
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json(), None
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误 {e.response.status_code}: {e.response.text}"
            print(f"Error: {error_msg}")
            return None, error_msg
    
    def poll_task_result(
        self,
        task_id: str,
        use_cn_endpoint: bool = True,
        max_retries: int = 30,
        retry_delay: int = 5,
        timeout: int = 10
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
        """
        轮询任务结果
        
        Returns:
            Tuple[任务数据, 错误信息, 重试次数]
        """
        base_url = self.get_base_url(use_cn_endpoint)
        endpoint = "/v1/draw/result"
        url = f"{base_url}{endpoint}"
        
        headers = self.get_headers()
        payload = {"id": task_id}
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=timeout)
                
                if response.status_code != 200:
                    print(f"请求失败，状态码: {response.status_code}, 重试 {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                
                result = response.json()
                
                if result.get('code') == 0:
                    data = result.get('data', {})
                    status = data.get('status', '')
                    progress = data.get('progress', 0)
                    
                    if status in ['succeeded', 'failed']:
                        return data, None, attempt + 1
                    else:
                        print(f"任务处理中: {progress}%, 状态: {status}, 重试 {attempt + 1}/{max_retries}")
                        time.sleep(retry_delay)
                        continue
                        
                elif result.get('code') == -22:
                    print(f"任务不存在, 重试 {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                    
                else:
                    error_msg = result.get('msg', '未知错误')
                    print(f"API错误: {error_msg}, 重试 {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                print(f"轮询错误: {str(e)}, 重试 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        return None, "轮询超时", max_retries