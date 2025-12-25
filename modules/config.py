# config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class APIMartConfig:
    """API Mart 配置"""
    # API 配置
    BASE_URL: str = "https://api.apimart.ai/v1/videos/generations"
    STATUS_URL: str = "https://api.apimart.ai/v1/tasks"
    IMAGES_BASE_URL: str = "https://api.apimart.ai/v1/images/generations"
    
    # 超时和重试
    TIMEOUT: int = 120
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    POLL_INTERVAL: int = 5
    MAX_POLL_TIME: int = 300
    
    # 文件配置
    MAX_TASK_HISTORY: int = 50
    
    # 图像配置
    MAX_IMAGE_SIDE: int = 1024
    MAX_VIDEO_UPLOAD_SIZE_MB: int = 100
    
    @classmethod
    def get_api_key(cls, api_key: str = "") -> str:
        """获取API密钥，优先使用传入的，其次使用环境变量"""
        key = (api_key or "").strip()
        if not key:
            key = os.getenv("SORA2_API_KEY", "").strip()
        return key
    
    @classmethod
    def get_headers(cls, api_key: str = "") -> Dict[str, str]:
        """获取请求头"""
        key = cls.get_api_key(api_key)
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
        }
    
    @classmethod
    def get_form_headers(cls, api_key: str = "") -> Dict[str, str]:
        """获取表单上传请求头"""
        key = cls.get_api_key(api_key)
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
        }