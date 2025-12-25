# response_parser.py
import json
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

class ResponseParser:
    """统一响应解析器"""
    
    @staticmethod
    def sanitize_url(url: str) -> Optional[str]:
        """清理URL"""
        if not isinstance(url, str):
            return None
        
        url = url.strip().strip("`\"'")
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            return url
        return None
    
    @staticmethod
    def extract_task_id(response: Dict[str, Any]) -> Optional[str]:
        """提取任务ID"""
        if not isinstance(response, dict):
            return None
        
        # 尝试多种可能的字段路径
        possible_paths = [
            ["task_id"],
            ["data", "task_id"],
            ["data", 0, "task_id"],  # data是列表的情况
            ["id"],
            ["result", "task_id"],
        ]
        
        for path in possible_paths:
            value = response
            for key in path:
                try:
                    if isinstance(value, dict):
                        value = value.get(key)
                    elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                        value = value[key]
                    else:
                        value = None
                        break
                except Exception:
                    value = None
                    break
            
            if isinstance(value, str) and value.strip():
                return value.strip()
        
        return None
    
    @staticmethod
    def extract_video_url(response: Any) -> Optional[str]:
        """提取视频URL"""
        if isinstance(response, str):
            return ResponseParser.sanitize_url(response)
        
        if not isinstance(response, dict):
            return None
        
        # 定义可能的URL字段
        url_fields = ["video_url", "url", "result_url", "output_url"]
        
        for field in url_fields:
            value = response.get(field)
            if isinstance(value, str):
                url = ResponseParser.sanitize_url(value)
                if url:
                    return url
            elif isinstance(value, list):
                for item in value:
                    url = ResponseParser.extract_video_url(item)
                    if url:
                        return url
        
        # 深度搜索
        def deep_search(obj):
            if isinstance(obj, str):
                url = ResponseParser.sanitize_url(obj)
                if url:
                    return url
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in url_fields or "url" in key.lower():
                        result = deep_search(value)
                        if result:
                            return result
                    else:
                        result = deep_search(value)
                        if result:
                            return result
            
            if isinstance(obj, list):
                for item in obj:
                    result = deep_search(item)
                    if result:
                        return result
            
            return None
        
        return deep_search(response)
    
    @staticmethod
    def extract_image_urls(response: Any) -> List[str]:
        """提取多个图像URL"""
        urls = set()
        
        def extract(obj):
            if isinstance(obj, str):
                url = ResponseParser.sanitize_url(obj)
                if url:
                    urls.add(url)
            
            elif isinstance(obj, dict):
                # 优先检查特定字段
                for field in ["image_url", "url", "image_urls"]:
                    value = obj.get(field)
                    if isinstance(value, str):
                        url = ResponseParser.sanitize_url(value)
                        if url:
                            urls.add(url)
                    elif isinstance(value, list):
                        for item in value:
                            extract(item)
                
                # 递归检查所有值
                for value in obj.values():
                    extract(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    extract(item)
        
        extract(response)
        return list(urls)
    
    @staticmethod
    def extract_error_message(response: Dict[str, Any]) -> str:
        """提取错误信息"""
        if not isinstance(response, dict):
            return str(response)
        
        # 检查常见错误字段
        error_fields = ["error", "message", "detail", "reason"]
        for field in error_fields:
            value = response.get(field)
            if value:
                if isinstance(value, dict):
                    return value.get("message", str(value))
                return str(value)
        
        return ""
    
    @staticmethod
    def extract_progress(response: Dict[str, Any]) -> Optional[int]:
        """提取进度百分比"""
        if not isinstance(response, dict):
            return None
        
        # 检查多种可能的进度字段
        progress_fields = [
            "progress", "percentage", "complete_percent", 
            "completion", "percent", "percent_complete"
        ]
        
        for field in progress_fields:
            value = response.get(field)
            if value is not None:
                try:
                    progress = int(float(value))
                    if 0 <= progress <= 100:
                        return progress
                except (ValueError, TypeError):
                    continue
        
        # 检查嵌套字段
        for nested in ["data", "output", "result", "status"]:
            if isinstance(response.get(nested), dict):
                for field in progress_fields:
                    value = response[nested].get(field)
                    if value is not None:
                        try:
                            progress = int(float(value))
                            if 0 <= progress <= 100:
                                return progress
                        except (ValueError, TypeError):
                            continue
        
        return None
    
    @staticmethod
    def extract_status(response: Dict[str, Any]) -> Optional[str]:
        """提取任务状态"""
        if not isinstance(response, dict):
            return None
        
        # 检查常见的状态字段
        status_fields = [
            "status", "state", "task_status", "job_status",
            "phase", "stage", "condition"
        ]
        
        for field in status_fields:
            value = response.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        
        # 检查嵌套字段
        for nested in ["data", "output", "result", "task", "job"]:
            if isinstance(response.get(nested), dict):
                for field in status_fields:
                    value = response[nested].get(field)
                    if isinstance(value, str) and value.strip():
                        return value.strip().lower()
        
        # 根据其他字段推断状态
        if response.get("error"):
            return "error"
        elif response.get("completed"):
            return "completed"
        elif response.get("succeeded"):
            return "succeeded"
        elif response.get("failed"):
            return "failed"
        
        return None
    
    @staticmethod
    def extract_task_info(response: Dict[str, Any]) -> Dict[str, Any]:
        """提取完整的任务信息"""
        if not isinstance(response, dict):
            return {}
        
        info = {
            "status": ResponseParser.extract_status(response),
            "progress": ResponseParser.extract_progress(response),
            "error": ResponseParser.extract_error_message(response),
            "video_url": ResponseParser.extract_video_url(response),
            "image_urls": ResponseParser.extract_image_urls(response),
            "task_id": ResponseParser.extract_task_id(response),
        }
        
        # 添加原始响应的其他有用字段
        for key in ["created_at", "updated_at", "estimated_completion", "duration"]:
            if key in response:
                info[key] = response[key]
        
        return info
    
    @staticmethod
    def extract_error_details(response: Dict[str, Any]) -> Dict[str, Any]:
        """提取详细的错误信息"""
        if not isinstance(response, dict):
            return {"error": str(response)}
        
        error_info = {
            "message": ResponseParser.extract_error_message(response),
            "code": None,
            "details": None,
            "suggestion": None,
        }
        
        # 提取错误代码
        for field in ["code", "error_code", "status_code", "error_code"]:
            if field in response:
                error_info["code"] = str(response[field])
                break
        
        # 提取详细信息
        for field in ["details", "detail", "error_details", "trace"]:
            if field in response:
                error_info["details"] = response[field]
                break
        
        # 检查嵌套错误字段
        if "error" in response and isinstance(response["error"], dict):
            error_obj = response["error"]
            for field in ["message", "code", "details", "suggestion"]:
                if field in error_obj and error_obj[field]:
                    error_info[field] = error_obj[field]
        
        return error_info