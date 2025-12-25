# api_client.py
import requests
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from .config import APIMartConfig

class APIMartClient:
    """API Mart 统一客户端"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.config = APIMartConfig
        
    def request_with_retry(
        self, 
        method: str, 
        url: str, 
        max_retries: int = None,
        retry_delay: int = None,
        **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """带重试的HTTP请求"""
        max_retries = max_retries or self.config.MAX_RETRIES
        retry_delay = retry_delay or self.config.RETRY_DELAY
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    timeout=self.config.TIMEOUT,
                    **kwargs
                )
                
                # 非5xx错误不重试
                if response.status_code < 500:
                    try:
                        return response.status_code, response.json()
                    except Exception:
                        return response.status_code, {"raw": response.text}
                
                if attempt == max_retries:
                    return response.status_code, {"raw": response.text}
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    return 0, {"error": str(e)}
            
            time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
        return 0, {"error": "Max retries exceeded"}
    
    def submit_video_generation(
        self, 
        payload: Dict[str, Any],
        use_multipart: bool = False,
        files: Dict = None
    ) -> Tuple[int, Dict[str, Any]]:
        """提交视频生成任务"""
        if use_multipart:
            return self.request_with_retry(
                "POST",
                self.config.BASE_URL,
                files=files,
                data=payload,
                headers=self.config.get_form_headers(self.api_key)
            )
        else:
            return self.request_with_retry(
                "POST",
                self.config.BASE_URL,
                json=payload,
                headers=self.config.get_headers(self.api_key)
            )
    
    def submit_image_generation(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """提交图像生成任务"""
        return self.request_with_retry(
            "POST",
            self.config.IMAGES_BASE_URL,
            json=payload,
            headers=self.config.get_headers(self.api_key)
        )
    
    def query_task(
        self, 
        task_id: str,
        poll_until_done: bool = False,
        max_poll_time: int = None
    ) -> Tuple[int, Dict[str, Any]]:
        """查询任务状态"""
        url = f"{self.config.STATUS_URL}/{task_id}"
        
        if not poll_until_done:
            return self.request_with_retry(
                "GET",
                url,
                headers=self.config.get_headers(self.api_key)
            )
        
        # 轮询直到任务完成
        max_poll_time = max_poll_time or self.config.MAX_POLL_TIME
        start_time = time.time()
        
        while time.time() - start_time < max_poll_time:
            status_code, response = self.request_with_retry(
                "GET",
                url,
                headers=self.config.get_headers(self.api_key)
            )
            
            if status_code != 200:
                return status_code, response
            
            status = self._extract_status(response)
            if status in ["completed", "failed", "cancelled"]:
                return status_code, response
            
            time.sleep(self.config.POLL_INTERVAL)
        
        return 408, {"error": "Polling timeout"}
    
    def _extract_status(self, response: Dict[str, Any]) -> Optional[str]:
        """从响应中提取状态"""
        if isinstance(response, dict):
            # 尝试多种可能的字段名
            for field in ["status", "state"]:
                if field in response:
                    return response[field]
            
            # 检查嵌套字段
            for nested in ["data", "output"]:
                if isinstance(response.get(nested), dict):
                    for field in ["status", "state"]:
                        if field in response[nested]:
                            return response[nested][field]
        
        return None