import os
import json
import time
import requests
import shutil
import re
import base64
import io
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from comfy.comfy_types import IO
from comfy_api.input import VideoInput
import folder_paths
from PIL import Image


# 分类名称
CATEGORY = "ZVNodes/apimart"

# 配置文件路径
NODE_DIR = os.path.dirname(__file__)
TASK_FILE = os.path.join(NODE_DIR, "apimart_task_history.json")

# API配置
API_CONFIG = {
    "video_base_url": "https://api.apimart.ai/v1/videos/generations",
    "image_base_url": "https://api.apimart.ai/v1/images/generations",
    "status_url": "https://api.apimart.ai/v1/tasks",
    "upload_token_url": "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH",
    "api_key": os.getenv("APIMART_API_KEY", ""),
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 2,
    "poll_interval": 5,
    "max_poll_time": 300,
}


# ==================== 任务管理函数 ====================
def _ensure_task_file():
    """确保任务文件存在"""
    if not os.path.exists(TASK_FILE):
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump({"tasks": []}, f, ensure_ascii=False, indent=2)

def _read_task_queue() -> List[Dict[str, Any]]:
    """读取任务队列"""
    _ensure_task_file()
    try:
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tasks", [])
    except Exception:
        return []

def _trim_task_list(tasks: List[Dict[str, Any]], max_keep: int = 100) -> List[Dict[str, Any]]:
    """修剪任务列表，保留最新的任务"""
    try:
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.get("saved_at", t.get("submitted_at", "1970-01-01 00:00:00")),
            reverse=True,
        )
        return sorted_tasks[:max_keep]
    except Exception:
        return tasks[:max_keep]

def _write_task_queue(tasks: List[Dict[str, Any]]):
    """写入任务队列"""
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)

def _save_task(task_id: str, task_type: str = "video", model: str = "", prompt: str = ""):
    """保存任务到历史记录"""
    tasks = _read_task_queue()
    tasks.append({
        "task_id": task_id,
        "task_type": task_type,
        "model": model,
        "prompt": prompt[:200],  # 只保存前200个字符
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "submitted"
    })
    tasks = _trim_task_list(tasks)
    _write_task_queue(tasks)

def _remove_task_by_id(task_id: str):
    """根据任务ID移除任务"""
    tasks = _read_task_queue()
    new_tasks = [t for t in tasks if t.get("task_id") != task_id]
    _write_task_queue(new_tasks)

def _update_task_status(task_id: str, status: str, result_url: str = ""):
    """更新任务状态"""
    tasks = _read_task_queue()
    for task in tasks:
        if task.get("task_id") == task_id:
            task["status"] = status
            if result_url:
                task["result_url"] = result_url
            task["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            break
    _write_task_queue(tasks)


# ==================== HTTP辅助函数 ====================
def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """生成请求头"""
    key = (api_key or "").strip() or API_CONFIG.get("api_key")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
    }

def _sanitize_url(s: str) -> Optional[str]:
    """清理URL字符串"""
    if not isinstance(s, str):
        return None
    s2 = s.strip().strip("`\"'")
    return s2 if s2.startswith("http") else None

def _extract_url_from_response(obj: Any, is_image: bool = False) -> Optional[str]:
    """从API响应中提取URL（支持视频和图片）"""
    if isinstance(obj, str):
        return _sanitize_url(obj)
    
    if isinstance(obj, dict):
        # 尝试常见字段
        for key in ("url", "video_url", "image_url", "result_url", "download_url"):
            v = obj.get(key)
            if isinstance(v, str):
                sv = _sanitize_url(v)
                if sv:
                    return sv
        
        # 尝试data字段
        data = obj.get("data")
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict):
                    # 检查status是否为success/completed
                    status = item.get("status", "").lower()
                    if status in ("success", "completed", "finished"):
                        for key in ("url", "video_url", "image_url"):
                            v = item.get(key)
                            if isinstance(v, str):
                                sv = _sanitize_url(v)
                                if sv:
                                    return sv
        
        # 尝试output字段
        output = obj.get("output")
        if isinstance(output, dict):
            for key in ("url", "video_url", "image_url"):
                v = output.get(key)
                if isinstance(v, str):
                    sv = _sanitize_url(v)
                    if sv:
                        return sv
        
        # 尝试results字段
        results = obj.get("results")
        if isinstance(results, list) and results:
            for item in results:
                if isinstance(item, dict):
                    for key in ("url", "video_url", "image_url"):
                        v = item.get(key)
                        if isinstance(v, str):
                            sv = _sanitize_url(v)
                            if sv:
                                return sv
    
    if isinstance(obj, list) and obj:
        for item in obj:
            url = _extract_url_from_response(item, is_image)
            if url:
                return url
    
    return None

def _extract_task_status(obj: Any) -> Tuple[str, str]:
    """从API响应中提取任务状态和消息"""
    status = "unknown"
    message = ""
    
    if isinstance(obj, dict):
        # 尝试提取状态
        for key in ("status", "state"):
            if key in obj:
                status = str(obj[key]).lower()
                break
        
        # 尝试从data中提取状态
        data = obj.get("data")
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict):
                    for key in ("status", "state"):
                        if key in item:
                            status = str(item[key]).lower()
                            break
        
        # 提取消息
        for key in ("message", "msg", "error"):
            if key in obj:
                message = str(obj[key])
                break
        
        # 如果是错误响应
        if "error" in obj:
            error_obj = obj["error"]
            if isinstance(error_obj, dict):
                message = error_obj.get("message", str(error_obj))
            else:
                message = str(error_obj)
    
    return status, message

def _query_task(task_id: str, api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """查询任务状态"""
    url = f"{API_CONFIG.get('status_url')}/{task_id}"
    timeout = API_CONFIG.get("timeout", 30)
    try:
        resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


# ==================== 图像处理函数 ====================
def _tensor_to_pil(img_any: Any):
    """将张量或数组转换为PIL图像"""
    if Image is None:
        return None
    
    try:
        # 如果是torch.Tensor
        if torch is not None and isinstance(img_any, torch.Tensor):
            t = img_any
            if t.dim() == 4:  # [B, H, W, C]
                t = t[0]
            t = t.detach().cpu().clamp(0, 1)
            arr = (t.numpy() * 255).astype("uint8")
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            elif arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
        
        # 如果是numpy数组
        if np is not None and isinstance(img_any, np.ndarray):
            arr = img_any
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            elif arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
        
        # 如果是PIL图像
        if hasattr(img_any, "save"):
            return img_any
            
        # 如果对象有to_pil方法
        if hasattr(img_any, "to_pil"):
            return img_any.to_pil()
            
        # 如果是字典结构
        if isinstance(img_any, dict):
            candidate = img_any.get("image") or (img_any.get("images")[0] if img_any.get("images") else None)
            if candidate:
                if hasattr(candidate, "save"):
                    return candidate
                elif hasattr(candidate, "to_pil"):
                    return candidate.to_pil()
                else:
                    return _tensor_to_pil(candidate)
                    
    except Exception as e:
        print(f"Error converting tensor to PIL: {e}")
    
    return None

def _image_to_temp_png(image: Any, max_side: int = 1024) -> Optional[str]:
    """将图像转换为临时PNG文件"""
    pil_img = _tensor_to_pil(image)
    if pil_img is None:
        return None
    
    try:
        # 转换为RGB
        if pil_img.mode not in ("RGB", "RGBA") and Image is not None:
            pil_img = pil_img.convert("RGB")
        
        # 调整大小
        if Image is not None:
            w, h = pil_img.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 保存到临时文件
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(tmp, "PNG", optimize=True)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"Error creating temp PNG: {e}")
        return None

def _upload_image_to_cdn(image: Any, api_key: Optional[str], max_side: int = 1024) -> Optional[str]:
    """上传图像到CDN"""
    temp_png = _image_to_temp_png(image, max_side)
    if not temp_png:
        return None
    
    try:
        headers = _headers(api_key)
        
        # 获取上传令牌
        token_res = requests.post(
            API_CONFIG.get("upload_token_url"),
            headers=headers,
            json={"sux": "png"},
            timeout=30
        )
        
        if token_res.status_code != 200:
            return None
        
        try:
            token_data = token_res.json().get("data") or {}
        except Exception:
            return None
        
        token = token_data.get("token")
        key = token_data.get("key")
        up_url = token_data.get("url")
        domain = token_data.get("domain")
        
        if not (token and key and up_url and domain):
            return None
        
        # 上传文件
        with open(temp_png, "rb") as f:
            up_resp = requests.post(
                up_url,
                data={"token": token, "key": key},
                files={"file": f},
                timeout=120
            )
        
        if up_resp.status_code not in (200, 201):
            return None
        
        # 构造公开URL
        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception as e:
        print(f"Error uploading to CDN: {e}")
        return None
    finally:
        try:
            if temp_png and os.path.exists(temp_png):
                os.unlink(temp_png)
        except Exception:
            pass


# ==================== 视频处理函数 ====================
def _video_to_temp_file(video: Any) -> Optional[str]:
    """将视频转换为临时文件"""
    try:
        # 如果是路径字符串
        if isinstance(video, str) and os.path.exists(video):
            ext = os.path.splitext(video)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(video, tmp.name)
            return tmp.name
        
        # 如果有path属性
        path = getattr(video, "path", None)
        if isinstance(path, str) and os.path.exists(path):
            ext = os.path.splitext(path)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(path, tmp.name)
            return tmp.name
        
        # 如果有save_to方法
        if hasattr(video, "save_to"):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            try:
                if video.save_to(tmp.name):
                    return tmp.name
            except Exception:
                pass
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _upload_video_to_cdn(video: Any, api_key: Optional[str]) -> Optional[str]:
    """上传视频到CDN"""
    temp_video = _video_to_temp_file(video)
    if not temp_video:
        return None
    
    try:
        headers = _headers(api_key)
        ext = os.path.splitext(temp_video)[1].lower().lstrip(".") or "mp4"
        
        # 获取上传令牌
        token_res = requests.post(
            API_CONFIG.get("upload_token_url"),
            headers=headers,
            json={"sux": ext},
            timeout=30
        )
        
        if token_res.status_code != 200:
            return None
        
        try:
            token_data = token_res.json().get("data") or {}
        except Exception:
            return None
        
        token = token_data.get("token")
        key = token_data.get("key")
        up_url = token_data.get("url")
        domain = token_data.get("domain")
        
        if not (token and key and up_url and domain):
            return None
        
        # 上传文件
        with open(temp_video, "rb") as f:
            up_resp = requests.post(
                up_url,
                data={"token": token, "key": key},
                files={"file": f},
                timeout=300
            )
        
        if up_resp.status_code not in (200, 201):
            return None
        
        # 构造公开URL
        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception as e:
        print(f"Error uploading video to CDN: {e}")
        return None
    finally:
        try:
            if temp_video and os.path.exists(temp_video):
                os.unlink(temp_video)
        except Exception:
            pass


# ==================== 下载函数 ====================
def _download_with_retries(url: str, target_path: str, max_retries: int = 3) -> Tuple[bool, str]:
    """带重试的下载函数"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
    backoff = 5.0
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=API_CONFIG.get("timeout", 30))
            resp.raise_for_status()
            
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True, f"下载成功: {target_path}"
        except Exception as e:
            if attempt == max_retries:
                return False, f"下载失败: {e}"
            time.sleep(backoff)
            backoff *= 2
    
    return False, "下载失败"


# ==================== API调用函数 ====================
def _submit_image_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交图像生成任务"""
    url = API_CONFIG.get("image_base_url")
    timeout = API_CONFIG.get("timeout", 30)
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}

def _submit_video_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交视频生成任务"""
    url = API_CONFIG.get("video_base_url")
    timeout = API_CONFIG.get("timeout", 30)
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


# ==================== VideoAdapter类 ====================
class VideoAdapterZV:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.width, self.height, self.fps = self._get_video_details(path)
    
    def _get_video_details(self, path: Optional[str]):
        try:
            if not path or not os.path.exists(path):
                return 1280, 720, 30
            
            # 尝试使用OpenCV获取视频信息
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps and fps > 0:
                        return width or 1280, height or 720, int(fps)
            except Exception:
                pass
            
            return 1280, 720, 30
        except Exception:
            return 1280, 720, 30
    
    def __repr__(self):
        return f"<VideoAdapterZV path={self.path} {self.width}x{self.height}@{self.fps}>"
    
    def get_dimensions(self):
        return self.width, self.height
    
    def save_to(self, output_path, **kwargs):
        try:
            if self.path and os.path.exists(self.path):
                shutil.copyfile(self.path, output_path)
                return True
            return False
        except Exception:
            return False


# ==================== 新增：SeeDream 4.0 图像生成节点 ====================
class SeeDream40Text2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "doubao-seedance-4-0",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"SeeDream 4.0 - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "doubao-seedance-4-0", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 新增：SeeDream 4.5 图像生成节点 ====================
class SeeDream45Text2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "doubao-seedance-4-5",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"SeeDream 4.5 - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "doubao-seedance-4-5", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 新增：Nano Banana Pro 图像生成节点 ====================
class NanoBananaProText2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "gemini-3-pro-image-preview",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"Nano Banana Pro - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "gemini-3-pro-image-preview", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 新增：通用任务轮询节点 ====================
class ApimartTaskPollingZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_poll_time": ("INT", {"default": 300, "min": 10, "max": 3600}),
                "task_type": (["auto", "image", "video"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "video_path", "report")
    CATEGORY = CATEGORY
    FUNCTION = "poll"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def poll(self, task_id: str, api_key: str, poll_interval: int, max_poll_time: int, task_type: str):
        if not task_id.strip():
            return (None, None, "错误: 未提供task_id")
        
        start_time = time.time()
        last_report = ""
        
        while time.time() - start_time < max_poll_time:
            # 查询任务状态
            code, body = _query_task(task_id, api_key)
            
            if code != 200:
                return (None, None, f"查询失败: HTTP {code}, 响应: {json.dumps(body, ensure_ascii=False)[:200]}")
            
            # 提取状态和消息
            status, message = _extract_task_status(body)
            
            # 检查任务是否完成
            if status in ("success", "completed", "finished", "succeeded"):
                # 提取结果URL
                result_url = _extract_url_from_response(body)
                
                if not result_url:
                    return (None, None, f"任务完成但未找到结果URL, 响应: {json.dumps(body, ensure_ascii=False)[:200]}")
                
                # 确定文件类型
                file_ext = os.path.splitext(result_url)[1].lower()
                is_image = file_ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
                is_video = file_ext in (".mp4", ".mov", ".avi", ".webm", ".mkv")
                
                if task_type == "auto":
                    actual_type = "image" if is_image else "video" if is_video else "unknown"
                else:
                    actual_type = task_type
                
                # 下载文件
                output_dir = folder_paths.get_output_directory()
                timestamp = int(time.time())
                
                if actual_type == "image" or is_image:
                    # 下载图片
                    filename = f"apimart_image_{task_id}_{timestamp}{file_ext}"
                    output_path = os.path.join(output_dir, filename)
                    
                    ok, msg = _download_with_retries(result_url, output_path)
                    if not ok:
                        return (None, None, f"图片下载失败: {msg}")
                    
                    # 转换为ComfyUI IMAGE格式
                    try:
                        pil_img = Image.open(output_path)
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        
                        # 转换为numpy数组
                        img_array = np.array(pil_img).astype(np.float32) / 255.0
                        
                        # 转换为torch张量
                        if torch is not None:
                            img_tensor = torch.from_numpy(img_array)[None, ...]
                        else:
                            img_tensor = img_array[None, ...]
                        
                        _update_task_status(task_id, "completed", result_url)
                        return (img_tensor, None, f"图片下载成功: {filename} | URL: {result_url}")
                    except Exception as e:
                        return (None, None, f"图片处理失败: {e}")
                
                elif actual_type == "video" or is_video:
                    # 下载视频
                    filename = f"apimart_video_{task_id}_{timestamp}{file_ext}"
                    output_path = os.path.join(output_dir, filename)
                    
                    ok, msg = _download_with_retries(result_url, output_path)
                    if not ok:
                        return (None, None, f"视频下载失败: {msg}")
                    
                    # video_adapter = VideoAdapterZV(output_path)
                    _update_task_status(task_id, "completed", result_url)
                    return (None, output_path, f"视频下载成功: {filename} | URL: {result_url}")
                
                else:
                    return (None, None, f"无法确定文件类型: {file_ext}")
            
            elif status in ("failed", "error", "canceled"):
                return (None, None, f"任务失败: {message or '未知错误'}")
            
            elif status in ("processing", "running", "submitted"):
                # 任务还在处理中，继续轮询
                elapsed = int(time.time() - start_time)
                report = f"任务处理中... 状态: {status}, 已等待: {elapsed}秒"
                if message:
                    report += f", 消息: {message}"
                
                if report != last_report:
                    print(report)
                    last_report = report
                
                time.sleep(poll_interval)
                continue
            
            else:
                # 未知状态
                elapsed = int(time.time() - start_time)
                report = f"未知状态: {status}, 已等待: {elapsed}秒"
                if message:
                    report += f", 消息: {message}"
                
                if report != last_report:
                    print(report)
                    last_report = report
                
                time.sleep(poll_interval)
        
        # 超时
        return (None, None, f"轮询超时 ({max_poll_time}秒), 最后状态: {status}, 消息: {message}")


# ==================== 原有的视频生成节点（保持兼容） ====================
class ApimartText2VideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "10"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        
        code, body = _submit_video_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video", model, prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


class ApimartImage2VideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "10"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, image, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        # 上传图片到CDN
        image_url = _upload_image_to_cdn(image, api_key)
        
        if not image_url:
            return ("错误: 图片上传失败", "")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
            "image_urls": [image_url],
        }
        
        code, body = _submit_video_generation(payload, api_key)
        
        # 如果使用image_urls失败，尝试使用url字段
        if code != 200:
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": int(duration),
                "url": image_url,
            }
            code, body = _submit_video_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video", model, prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


class ApimartDownloadSavedTaskVideoZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def run(self, api_key: str, task_id: str = ""):
        manual_id = (task_id or "").strip()
        if manual_id:
            selected_task_id = manual_id
        else:
            tasks = _read_task_queue()
            if not tasks:
                return (VideoAdapterZV(None), "没有已保存的任务ID")
            selected_task_id = tasks[0].get("task_id")
        
        # 使用轮询逻辑下载
        polling_node = ApimartTaskPollingZV()
        image, video, report = polling_node.poll(selected_task_id, api_key, 5, 300, "auto")
        
        if video is not None:
            return (video, report)
        else:
            return (VideoAdapterZV(None), report)


# ==================== 视频Remix节点 ====================
def _submit_remix_by_video_id(video_id: str, prompt: str, model: str, duration: int, 
                            api_key: Optional[str], aspect_ratio: str = "16:9") -> Tuple[int, Dict[str, Any]]:
    """通过视频ID提交Remix任务"""
    url = f"https://api.apimart.ai/v1/videos/{video_id}/remix"
    timeout = API_CONFIG.get("timeout", 30)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


class ApimartRemixByTaskIdSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "15"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, task_id: str, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        vid = (task_id or "").strip()
        if not vid:
            return ("错误: 未提供有效的task_id", "")
        
        final_duration = int(duration)
        code, body = _submit_remix_by_video_id(vid, prompt, model, final_duration, api_key, aspect_ratio=aspect_ratio)
        
        # 提取新任务ID
        new_task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                new_task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                new_task_id = data.get("task_id")
            new_task_id = new_task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if new_task_id:
            _save_task(new_task_id, "video_remix", model, prompt)
            report += f" | task_id: {new_task_id} 已保存"
        else:
            # 若服务端返回消息，附加到回执里
            msg = body.get("message") if isinstance(body, dict) else None
            if msg:
                report += f" | {msg}"
            else:
                report += " | 未返回 task_id"
        
        return (report, new_task_id or "")


# ==================== 视频Remix通过视频节点 ====================
def _submit_remix_by_url(video_url: str, prompt: str, model: str, duration: int, 
                        api_key: Optional[str], aspect_ratio: str = "16:9") -> Tuple[int, Dict[str, Any]]:
    """通过视频URL提交Remix任务"""
    url = "https://api.apimart.ai/v1/videos/remix"
    timeout = API_CONFIG.get("timeout", 30)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "url": video_url,
    }
    
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


class ApimartRemixVideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "15"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, video, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        final_duration = int(duration)
        
        # 先上传视频到CDN
        video_url = _upload_video_to_cdn(video, api_key)
        if not video_url:
            return ("错误: 视频上传失败", "")
        
        # 通过URL提交Remix
        code, body = _submit_remix_by_url(video_url, prompt, model, final_duration, api_key, aspect_ratio)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video_remix", model, prompt)
            report += f" | task_id: {task_id} 已保存"
        else:
            report += " | 未返回 task_id"
        
        return (report, task_id or "")

# ==================== 节点映射 ====================
NODE_CLASS_MAPPINGS = {
    # 原有节点（已添加ZV）
    "ApimartText2VideoSubmitZV": ApimartText2VideoSubmitZV,
    "ApimartImage2VideoSubmitZV": ApimartImage2VideoSubmitZV,
    "ApimartDownloadSavedTaskVideoZV": ApimartDownloadSavedTaskVideoZV,
    
    # 新增图像生成节点（已添加ZV）
    "SeeDream40Text2ImageSubmitZV": SeeDream40Text2ImageSubmitZV,
    "SeeDream45Text2ImageSubmitZV": SeeDream45Text2ImageSubmitZV,
    "NanoBananaProText2ImageSubmitZV": NanoBananaProText2ImageSubmitZV,
    
    # 新增通用轮询节点（已添加ZV）
    "ApimartTaskPollingZV": ApimartTaskPollingZV,

    # 视频Remix节点
    "ApimartRemixByTaskIdSubmitZV": ApimartRemixByTaskIdSubmitZV,
    "ApimartRemixVideoSubmitZV": ApimartRemixVideoSubmitZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 原有节点（已添加ZV）
    "ApimartText2VideoSubmitZV": "文生视频提交任务ZV",
    "ApimartImage2VideoSubmitZV": "图生视频提交任务ZV",
    "ApimartDownloadSavedTaskVideoZV": "下载已保存任务视频ZV",
    
    # 新增图像生成节点（已添加ZV）
    "SeeDream40Text2ImageSubmitZV": "SeeDream 4.0 文生图ZV",
    "SeeDream45Text2ImageSubmitZV": "SeeDream 4.5 文生图ZV",
    "NanoBananaProText2ImageSubmitZV": "Nano Banana Pro 文生图ZV",
    
    # 新增通用轮询节点（已添加ZV）
    "ApimartTaskPollingZV": "通用任务轮询下载ZV",

    # 视频Remix节点
    "ApimartRemixByTaskIdSubmitZV": "通过TaskID视频RemixZV",
    "ApimartRemixVideoSubmitZV": "视频Remix提交任务ZV",
}